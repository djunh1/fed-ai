"""Timestamped parquet snapshots: `<prefix><YYYYMMDD_HHMMSS>.parquet`, newest
by filename wins (ADR 0002). Knows nothing about what the columns mean."""

import datetime
import os
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_STAMP = re.compile(r"(\d{8}_\d{6})\.parquet$")


def newest_snapshot(directory: Path, prefix: str) -> Path | None:
    """Latest snapshot by the timestamp in its name, not mtime. None if none."""
    if not directory.is_dir():
        return None
    stamped = [
        (match.group(1), path)
        for path in directory.glob(f"{prefix}*.parquet")
        if (match := _STAMP.search(path.name)) and path.name == f"{prefix}{match.group(1)}.parquet"
    ]
    return max(stamped)[1] if stamped else None


def conform(frame: pd.DataFrame, schema: Mapping[str, str]) -> pd.DataFrame:
    """Schema's columns, in its order, cast to its dtypes.

    Missing columns become nulls, except bool, which becomes False (a null
    cast to bool is True).
    """
    extra = [c for c in frame.columns if c not in schema]
    if extra:
        logger.warning(f"Dropping columns not in schema: {extra}")
    out = pd.DataFrame(index=frame.index)
    for column, dtype in schema.items():
        if column in frame.columns:
            values = frame[column]
        else:
            values = pd.Series(False if dtype == "bool" else None, index=frame.index, dtype=object)
        out[column] = values.astype(dtype)
    return out.reset_index(drop=True)


def read_snapshot(path: Path, schema: Mapping[str, str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return conform(frame, schema) if schema else frame


def write_snapshot(
    frame: pd.DataFrame,
    directory: Path,
    prefix: str,
    *,
    now: datetime.datetime | None = None,
) -> Path:
    """Write via a .tmp then rename, so readers never see a partial file.

    Refuses to overwrite: two runs in the same second is a bug, not an update.
    """
    stamp = (now or datetime.datetime.now()).strftime(TIMESTAMP_FORMAT)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{prefix}{stamp}.parquet"
    if final.exists():
        raise FileExistsError(f"Snapshot already exists: {final}")

    tmp = final.with_name(final.name + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return final
