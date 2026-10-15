import datetime
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger
from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.metadata.metadata_types import Path as MetaPath
from zenml.metadata.metadata_types import StorageSize

from llm_engineering.application import fmp_enriched, fmp_snapshot, price_action
from llm_engineering.application.api.fmp import FmpError
from llm_engineering.application.api.fmp_api import FmpEarningsApi
from llm_engineering.infrastructure.parquet import (
    newest_snapshot,
    read_snapshot,
    write_snapshot,
)

CONSECUTIVE_FAILURE_ABORT = 10


@step
def enrich_fmp_price_action(
    snapshot_path: str,
    enriched_dir: str = "data/fmp_enriched",
    incremental: bool = False,
    price_action_from: str = "2016-01-01",
    price_action_to: str | None = None,
    limit: int | None = None,
    max_failure_pct: float = 5.0,
) -> Annotated[str, ArtifactConfig(name="fmp_enriched_snapshot", tags=["fmp", "price_action", "parquet"])]:
    """Join price action onto the earnings snapshot and write an enriched snapshot (ADR 0008).

    Incremental re-uses priced rows from the newest enriched file. Writes
    nothing if more than `max_failure_pct` of tickers fail.
    """
    started = time.monotonic()
    directory = Path(enriched_dir)
    priced_at = datetime.datetime.now(datetime.timezone.utc)
    today = priced_at.date()
    price_from = datetime.date.fromisoformat(price_action_from)
    price_to = datetime.date.fromisoformat(price_action_to) if price_action_to else None

    earnings = read_snapshot(Path(snapshot_path), fmp_snapshot.SCHEMA)
    baseline_path = newest_snapshot(directory, fmp_enriched.PREFIX) if incremental else None
    baseline = fmp_enriched.prices_from_enriched(
        read_snapshot(baseline_path, fmp_enriched.SCHEMA) if baseline_path else None
    )

    plan = fmp_enriched.select_rows_to_price(
        earnings,
        baseline,
        today=today,
        price_from=price_from,
        price_to=price_to,
        lookforward_days=price_action.lookforward_days(),
    )
    tickers = list(plan.rows)[:limit] if limit is not None else list(plan.rows)
    logger.info(
        f"Pricing {sum(len(plan.rows[t]) for t in tickers)} rows across {len(tickers)} tickers ({dict(plan.reasons)})"
    )

    api = FmpEarningsApi()
    fresh: list[dict] = []
    succeeded: list[str] = []
    no_prices: list[str] = []
    failed: dict[str, str] = {}
    consecutive = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            rows = api.fetch_price_action(ticker, plan.rows[ticker], today=today)
        except FmpError as exc:
            failed[ticker] = f"FmpError:{exc.status}"
            consecutive += 1
            if consecutive >= CONSECUTIVE_FAILURE_ABORT:
                raise RuntimeError(
                    f"{consecutive} consecutive FMP failures (last: {failed[ticker]}). No snapshot written."
                ) from None
            continue
        consecutive = 0
        (succeeded if rows else no_prices).append(ticker)
        fresh.extend(rows)
        if i % 100 == 0:
            logger.info(f"{i}/{len(tickers)} tickers, {len(failed)} failed")

    failure_pct = 100 * len(failed) / max(len(tickers), 1)
    if failure_pct > max_failure_pct:
        raise RuntimeError(
            f"{len(failed)}/{len(tickers)} tickers failed ({failure_pct:.1f}% > {max_failure_pct}%). "
            "No snapshot written."
        )

    prices = fmp_enriched.merge_prices(baseline, fmp_enriched.price_rows_to_frame(fresh, priced_at=priced_at))
    enriched = fmp_enriched.join(earnings, prices, price_from=price_from, price_to=price_to)
    path = write_snapshot(enriched, directory, fmp_enriched.PREFIX, now=priced_at)
    logger.info(f"Wrote {path} ({len(enriched)} rows)")

    get_step_context().add_output_metadata(
        output_name="fmp_enriched_snapshot",
        metadata={
            "query": {
                "incremental": incremental,
                "price_action_from": price_action_from,
                "price_action_to": price_action_to,
                "limit": limit,
                "max_failure_pct": max_failure_pct,
                "earnings_snapshot": Path(snapshot_path).name,
                "baseline": baseline_path.name if baseline_path else None,
            },
            "selection": {
                **{r: plan.reasons.get(r, 0) for r in ("no_baseline", "unpriced", "incomplete")},
                "kept_rows": plan.kept,
                "tickers_to_price": len(tickers),
            },
            "retrieved": {
                "succeeded": len(succeeded),
                "no_prices": len(no_prices),
                "no_prices_sample": sorted(no_prices)[:20],
                "failed": len(failed),
                "failed_by_reason": dict(Counter(failed.values())),
                "failed_sample": sorted(failed)[:20],
                "rows_priced": len(fresh),
                "elapsed_s": round(time.monotonic() - started, 1),
            },
            "snapshot": _snapshot_stats(enriched, path, today),
            "day_0_check": _day_0_check_stats(enriched),
        },
    )
    return str(path.resolve())


SUSPECT_SAMPLE = 50


def _day_0_check_stats(enriched: pd.DataFrame) -> dict:
    """Rows whose neighbour session gapped harder than day 0. Informational."""
    check = enriched["day_0_check"]
    checked = int(check.notna().sum())
    suspect = enriched[check.isin(["late", "early"])].sort_values("date", ascending=False)
    return {
        "checked": checked,
        "unchecked": int((check.isna() & (enriched["anchor_found"] == True)).sum()),  # noqa: E712
        "ok": int((check == "ok").sum()),
        "late": int((check == "late").sum()),
        "early": int((check == "early").sum()),
        "suspect_pct": round(100 * len(suspect) / checked, 2) if checked else None,
        "suspect_tickers": int(suspect["ticker"].nunique()),
        "suspect_sample": [
            f"{r.ticker} {r.date.date()} {r.time} {r.day_0_check}" for r in suspect.head(SUSPECT_SAMPLE).itertuples()
        ],
    }


def _snapshot_stats(enriched: pd.DataFrame, path: Path, today: datetime.date) -> dict:
    anchor = enriched["anchor_found"]
    past = enriched["date"] <= pd.Timestamp(today)
    return {
        "file": MetaPath(str(path.resolve())),
        "size": StorageSize(path.stat().st_size),
        "rows": len(enriched),
        "tickers": int(enriched["ticker"].nunique()),
        "earliest": str(enriched["date"].min().date()) if len(enriched) else None,
        "latest": str(enriched["date"].max().date()) if len(enriched) else None,
        "anchor_found": int((anchor == True).sum()),  # noqa: E712
        "anchor_missing": int((anchor == False).sum()),  # noqa: E712
        "unpriced_past_rows": int((anchor.isna() & past).sum()),
        "future_rows": int((~past).sum()),
        "complete_rows": int(enriched[price_action.LONGEST_HORIZON_COLUMN].notna().sum()),
        "null_market_cap": int((enriched["market_capitalization"].isna() & (anchor == True)).sum()),  # noqa: E712
        "null_sector": int((enriched["sector"].isna() & anchor.notna()).sum()),
    }
