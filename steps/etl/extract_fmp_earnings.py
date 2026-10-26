import datetime
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.metadata.metadata_types import Path as MetaPath
from zenml.metadata.metadata_types import StorageSize

from llm_engineering.application import fmp_snapshot
from llm_engineering.application.api.fmp import FmpError
from llm_engineering.application.api.fmp_api import FmpEarningsApi
from llm_engineering.application.api.fmp_models import EarningsRecord
from llm_engineering.infrastructure.parquet import newest_snapshot, read_snapshot, write_snapshot

#: A bad key or an outage fails every call; stop instead of burning the run.
CONSECUTIVE_FAILURE_ABORT = 10


@step
def extract_fmp_earnings(
    symbols: list[str],
    data_dir: str = "data/fmp_earnings",
    incremental: bool = False,
    stale_after_days: int = 75,
    limit: int | None = None,
    max_failure_pct: float = 5.0,
) -> Annotated[str, ArtifactConfig(name="fmp_earnings_snapshot", tags=["fmp", "earnings", "parquet"])]:
    """Fetch earnings for the symbols that need it and write a full snapshot (ADR 0002, 0004).

    Writes nothing if more than `max_failure_pct` of fetches fail, so the
    previous snapshot stays newest.
    """
    started = time.monotonic()
    directory = Path(data_dir)  # str from YAML
    ingested_at = datetime.datetime.now(datetime.timezone.utc)
    today = ingested_at.date()

    baseline_path = newest_snapshot(directory, fmp_snapshot.PREFIX) if incremental else None
    baseline = read_snapshot(baseline_path, fmp_snapshot.SCHEMA) if baseline_path else None

    plan = fmp_snapshot.select_symbols_to_refresh(
        baseline, symbols, today=today, stale_after_days=stale_after_days
    )
    to_fetch = plan.refresh[:limit] if limit is not None else plan.refresh
    logger.info(f"Refreshing {len(to_fetch)} of {len(symbols)} symbols ({Counter(plan.reasons.values())})")

    api = FmpEarningsApi()
    fetched: list[EarningsRecord] = []
    succeeded: set[str] = set()
    empty: list[str] = []
    failed: dict[str, str] = {}
    consecutive = 0

    for i, symbol in enumerate(to_fetch, 1):
        try:
            records = api.fetch_earnings(symbol)
        except FmpError as exc:
            failed[symbol] = f"FmpError:{exc.status}"  # never str(exc) here
            consecutive += 1
            if consecutive >= CONSECUTIVE_FAILURE_ABORT:
                raise RuntimeError(
                    f"{consecutive} consecutive FMP failures (last: {failed[symbol]}); "
                    "bad key or outage. No snapshot written."
                ) from None
            continue
        consecutive = 0
        if records:
            succeeded.add(symbol)
            fetched.extend(records)
        else:
            empty.append(symbol)
        if i % 100 == 0:
            logger.info(f"{i}/{len(to_fetch)} symbols, {len(failed)} failed")

    failure_pct = 100 * len(failed) / max(len(to_fetch), 1)
    if failure_pct > max_failure_pct:
        raise RuntimeError(
            f"{len(failed)}/{len(to_fetch)} symbols failed ({failure_pct:.1f}% > {max_failure_pct}%). "
            "No snapshot written."
        )

    fresh = fmp_snapshot.records_to_frame(fetched, ingested_at=ingested_at)
    merged = fmp_snapshot.merge_snapshot(baseline, fresh, succeeded=succeeded)
    path = write_snapshot(merged, directory, fmp_snapshot.PREFIX, now=ingested_at)
    logger.info(f"Wrote {path} ({len(merged)} rows)")

    get_step_context().add_output_metadata(
        output_name="fmp_earnings_snapshot",
        metadata=_get_metadata(
            incremental=incremental,
            stale_after_days=stale_after_days,
            limit=limit,
            max_failure_pct=max_failure_pct,
            universe=len(symbols),
            baseline_path=baseline_path,
            plan=plan,
            to_fetch=len(to_fetch),
            fetched=fetched,
            succeeded=succeeded,
            empty=empty,
            failed=failed,
            elapsed=time.monotonic() - started,
            path=path,
            merged=merged,
            today=today,
        ),
    )
    return str(path.resolve())


def _get_metadata(*, plan, merged, path: Path, fetched, failed: dict[str, str], today, **run) -> dict:
    reasons = Counter(plan.reasons.values())
    dates = merged["date"]
    return {
        "query": {
            "incremental": run["incremental"],
            "stale_after_days": run["stale_after_days"],
            "limit": run["limit"],
            "max_failure_pct": run["max_failure_pct"],
            "universe": run["universe"],
            "baseline": run["baseline_path"].name if run["baseline_path"] else None,
        },
        "selection": {
            **{reason: reasons.get(reason, 0) for reason in ("no_baseline", "new", "stale", "unresolved")},
            "kept": len(plan.kept),
            "dropped_from_universe": len(plan.dropped),
            "to_fetch": run["to_fetch"],
        },
        "retrieved": {
            "succeeded": len(run["succeeded"]),
            "empty": len(run["empty"]),
            "failed": len(failed),
            "failed_by_reason": dict(Counter(failed.values())),
            "failed_sample": sorted(failed)[:20],
            "rows_fetched": len(fetched),
            "elapsed_s": round(run["elapsed"], 1),
        },
        "snapshot": {
            "file": MetaPath(str(path.resolve())),
            "size": StorageSize(path.stat().st_size),
            "rows": len(merged),
            "symbols": int(merged["ticker"].nunique()),
            "earliest": str(dates.min().date()) if len(dates) else None,
            "latest": str(dates.max().date()) if len(dates) else None,
            "date_corrected_rows": int(merged["date_corrected"].sum()),
            "future_rows": int((dates > str(today)).sum()),
        },
    }
