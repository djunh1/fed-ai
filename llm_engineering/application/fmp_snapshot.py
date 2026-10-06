"""Earnings snapshot rules: schema, refresh selection, merge (ADR 0004).
File IO lives in infrastructure/parquet.py."""

import datetime
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from llm_engineering.application.api.fmp_models import OUTPUT_COLUMNS, SNAPSHOT_COLUMNS, EarningsRecord
from llm_engineering.infrastructure.parquet import conform

PREFIX = "earnings_"

_DTYPES = {
    "ticker": "string",
    "date": "datetime64[ns]",
    "time": "string",
    "timing_label": "string",
    "date_corrected": "bool",
    "fiscal_date_ending": "datetime64[ns]",
    "ingested_at": "datetime64[ns, UTC]",
}
#: Column -> dtype, in SNAPSHOT_COLUMNS order. Anything unlisted is float64.
SCHEMA: dict[str, str] = {c: _DTYPES.get(c, "float64") for c in SNAPSHOT_COLUMNS}


def empty_frame() -> pd.DataFrame:
    return conform(pd.DataFrame(), SCHEMA)


def records_to_frame(records: Iterable[EarningsRecord], *, ingested_at: datetime.datetime) -> pd.DataFrame:
    rows = [{c: getattr(r, c) for c in OUTPUT_COLUMNS} for r in records]
    if not rows:
        return empty_frame()
    frame = pd.DataFrame(rows)
    frame["ingested_at"] = pd.Timestamp(ingested_at).tz_convert("UTC")
    return conform(frame, SCHEMA)


@dataclass(frozen=True)
class RefreshPlan:
    refresh: list[str]  # universe order, so `limit` takes the largest first
    reasons: dict[str, str]  # symbol -> "new" | "stale" | "unresolved" | "no_baseline"
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # in baseline, not in universe; rows kept


def select_symbols_to_refresh(
    baseline: pd.DataFrame | None,
    universe: list[str],
    *,
    today: datetime.date,
    stale_after_days: int,
) -> RefreshPlan:
    """Which symbols an incremental run fetches. No baseline means all of them.

    stale: latest stored date older than `stale_after_days`.
    unresolved: a row inside that window is past but has no eps_actual.
    Older null rows are permanent FMP gaps and are ignored.
    """
    ordered = list(dict.fromkeys(universe))
    if baseline is None or baseline.empty:
        return RefreshPlan(refresh=ordered, reasons=dict.fromkeys(ordered, "no_baseline"))

    now = pd.Timestamp(today)
    window_start = now - pd.Timedelta(days=stale_after_days)
    latest = baseline.groupby("ticker")["date"].max()
    pending = baseline[(baseline["date"] < now) & (baseline["date"] >= window_start) & baseline["eps_actual"].isna()]
    unresolved = set(pending["ticker"])

    refresh, reasons, kept = [], {}, []
    for symbol in ordered:
        if symbol not in latest.index:
            reason = "new"
        elif (now - latest[symbol]).days > stale_after_days:
            reason = "stale"
        elif symbol in unresolved:
            reason = "unresolved"
        else:
            kept.append(symbol)
            continue
        refresh.append(symbol)
        reasons[symbol] = reason

    dropped = sorted(set(latest.index) - set(ordered))
    return RefreshPlan(refresh=refresh, reasons=reasons, kept=kept, dropped=dropped)


def merge_snapshot(baseline: pd.DataFrame | None, fresh: pd.DataFrame, *, succeeded: set[str]) -> pd.DataFrame:
    """Baseline with every `succeeded` symbol replaced wholesale by `fresh`.

    Failed or empty symbols keep their baseline rows. (ticker, date) is unique;
    fresh wins ties. Newest first, then ticker.
    """
    parts = [fresh]
    if baseline is not None and not baseline.empty:
        parts.insert(0, baseline[~baseline["ticker"].isin(succeeded)])
    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_frame()

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(["ticker", "date"], keep="last")
    merged = merged.sort_values(["date", "ticker"], ascending=[False, True], kind="stable")
    return conform(merged, SCHEMA)
