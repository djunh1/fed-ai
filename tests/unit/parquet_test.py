"""Snapshot file IO against tmp_path."""

import datetime
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from llm_engineering.infrastructure import parquet
from llm_engineering.infrastructure.parquet import conform, newest_snapshot, read_snapshot, write_snapshot

PREFIX = "earnings_"
SCHEMA = {
    "ticker": "string",
    "date": "datetime64[ns]",
    "flag": "bool",
    "value": "float64",
    "ingested_at": "datetime64[ns, UTC]",
}


def _at(text: str) -> datetime.datetime:
    return datetime.datetime.strptime(text, "%Y%m%d_%H%M%S")


def _frame(rows: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(rows)],
            "date": pd.date_range("2026-01-01", periods=rows),
            "flag": [i % 2 == 0 for i in range(rows)],
            "value": [float(i) if i else None for i in range(rows)],
            "ingested_at": pd.Timestamp("2026-10-06 12:00", tz="UTC"),
        }
    )
    return conform(frame, SCHEMA)


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path


# --- newest_snapshot --------------------------------------------------------


def test_newest_none_for_missing_or_empty_dir(tmp_path: Path) -> None:
    assert newest_snapshot(tmp_path / "nope", PREFIX) is None
    assert newest_snapshot(tmp_path, PREFIX) is None


def test_newest_by_filename_not_mtime(tmp_path: Path) -> None:
    newer = _touch(tmp_path, "earnings_20261006_120000.parquet")
    older = _touch(tmp_path, "earnings_20261005_120000.parquet")
    future = time.time() + 1000
    os.utime(older, (future, future))

    assert newest_snapshot(tmp_path, PREFIX) == newer


@pytest.mark.parametrize(
    "decoy",
    [
        "earnings_20991231_235959.parquet.tmp",
        "prices_20991231_235959.parquet",
        "earnings_latest.parquet",
        "earnings_20991231_235959_copy.parquet",
        "old_earnings_20991231_235959.parquet",
    ],
)
def test_newest_ignores_decoys(tmp_path: Path, decoy: str) -> None:
    real = _touch(tmp_path, "earnings_20261006_120000.parquet")
    _touch(tmp_path, decoy)

    assert newest_snapshot(tmp_path, PREFIX) == real


# --- conform ----------------------------------------------------------------


def test_conform_orders_and_casts() -> None:
    frame = pd.DataFrame({"value": [1], "ticker": ["A"], "flag": [True], "date": ["2026-01-01"], "ingested_at": [pd.Timestamp("2026-01-01", tz="UTC")]})

    out = conform(frame, SCHEMA)

    assert list(out.columns) == list(SCHEMA)
    assert {c: str(t) for c, t in out.dtypes.items()} == SCHEMA


def test_conform_adds_missing_as_null_and_drops_extra() -> None:
    frame = pd.DataFrame({"ticker": ["A"], "junk": [1]})

    out = conform(frame, SCHEMA)

    assert "junk" not in out.columns
    assert out["value"].isna().all()
    assert out["date"].isna().all()
    assert out["ingested_at"].isna().all()


def test_conform_missing_bool_is_false_not_true() -> None:
    out = conform(pd.DataFrame({"ticker": ["A", "B"]}), SCHEMA)

    assert out["flag"].tolist() == [False, False]


def test_conform_empty_frame_keeps_schema() -> None:
    out = conform(pd.DataFrame(), SCHEMA)

    assert out.empty
    assert {c: str(t) for c, t in out.dtypes.items()} == SCHEMA


# --- write + read -----------------------------------------------------------


def test_write_names_file_from_timestamp(tmp_path: Path) -> None:
    path = write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_141500"))

    assert path == tmp_path / "earnings_20261006_141500.parquet"


def test_write_creates_directory(tmp_path: Path) -> None:
    path = write_snapshot(_frame(), tmp_path / "a" / "b", PREFIX, now=_at("20261006_141500"))

    assert path.exists()


def test_write_leaves_no_tmp(tmp_path: Path) -> None:
    write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_141500"))

    assert list(tmp_path.glob("*.tmp")) == []


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_141500"))

    with pytest.raises(FileExistsError):
        write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_141500"))


def test_failed_write_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(parquet.os, "replace", boom)

    with pytest.raises(OSError):
        write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_141500"))
    assert list(tmp_path.iterdir()) == []


def test_round_trip_preserves_values_and_dtypes(tmp_path: Path) -> None:
    frame = _frame()

    back = read_snapshot(write_snapshot(frame, tmp_path, PREFIX, now=_at("20261006_141500")), SCHEMA)

    pd.testing.assert_frame_equal(back, frame)


def test_round_trip_without_schema_still_typed(tmp_path: Path) -> None:
    """pyarrow stores pandas dtypes, so even a schema-less read keeps them."""
    back = read_snapshot(write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_141500")))

    assert {c: str(t) for c, t in back.dtypes.items()} == SCHEMA


def test_round_trip_empty_frame(tmp_path: Path) -> None:
    empty = conform(pd.DataFrame(), SCHEMA)

    back = read_snapshot(write_snapshot(empty, tmp_path, PREFIX, now=_at("20261006_141500")), SCHEMA)

    pd.testing.assert_frame_equal(back, empty)


def test_newest_finds_what_write_wrote(tmp_path: Path) -> None:
    write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261005_090000"))
    second = write_snapshot(_frame(), tmp_path, PREFIX, now=_at("20261006_090000"))

    assert newest_snapshot(tmp_path, PREFIX) == second
