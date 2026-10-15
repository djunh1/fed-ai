"""Enriched snapshot: earnings rows from `price_from` on, joined to price
action on (ticker, date, time) (ADR 0008). Prices are kept per row and
re-used across runs; earnings columns always come from the latest snapshot."""

import datetime
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from llm_engineering.application import fmp_snapshot
from llm_engineering.application.api.fmp_models import OUTPUT_COLUMNS
from llm_engineering.application.price_action import LONGEST_HORIZON_COLUMN, PRICE_ACTION_COLUMNS
from llm_engineering.infrastructure.parquet import conform

PREFIX = "enriched_"
KEY = ["ticker", "date", "time"]

_PRICE_DTYPES = {
    "date_0d": "datetime64[ns]",
    "anchor_found": "boolean",  # nullable: NA means never priced
    "day_0_check": "string",
    "sector": "string",
    "industry": "string",
}
_PRICE_TYPES = {c: _PRICE_DTYPES.get(c, "float64") for c in PRICE_ACTION_COLUMNS}

#: Price table: key + price action + when it was computed.
PRICE_SCHEMA: dict[str, str] = {
    **{c: fmp_snapshot.SCHEMA[c] for c in KEY},
    **_PRICE_TYPES,
    "priced_at": "datetime64[ns, UTC]",
}

#: The published file: earnings columns, price action, then both timestamps.
SCHEMA: dict[str, str] = {
    **{c: fmp_snapshot.SCHEMA[c] for c in OUTPUT_COLUMNS},
    **_PRICE_TYPES,
    "ingested_at": fmp_snapshot.SCHEMA["ingested_at"],
    "priced_at": "datetime64[ns, UTC]",
}


def price_rows_to_frame(rows: Iterable[dict], *, priced_at: datetime.datetime) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if not frame.empty:
        frame["priced_at"] = pd.Timestamp(priced_at).tz_convert("UTC")
    return conform(frame, PRICE_SCHEMA)


def prices_from_enriched(enriched: pd.DataFrame | None) -> pd.DataFrame | None:
    """The price table held inside a previous enriched snapshot."""
    if enriched is None:
        return None
    return conform(enriched[list(PRICE_SCHEMA)], PRICE_SCHEMA)


@dataclass(frozen=True)
class PricePlan:
    rows: dict[str, list[tuple[datetime.date, str | None]]]  # ticker -> keys to price
    reasons: Counter = field(default_factory=Counter)  # per row: no_baseline | unpriced | incomplete
    kept: int = 0  # rows re-used from the baseline


def select_rows_to_price(
    earnings: pd.DataFrame,
    baseline: pd.DataFrame | None,
    *,
    today: datetime.date,
    price_from: datetime.date,
    lookforward_days: int,
    price_to: datetime.date | None = None,
) -> PricePlan:
    """Rows dated price_from..today (and ..price_to) that are unpriced, or recent and not yet complete.

    Incomplete rows older than `lookforward_days` are final (e.g. delisted) and kept.
    Future rows are skipped: no session to anchor on yet.
    """
    now = pd.Timestamp(today)
    last = min(now, pd.Timestamp(price_to)) if price_to else now
    eligible = earnings[(earnings["date"] >= pd.Timestamp(price_from)) & (earnings["date"] <= last)][KEY]

    if baseline is None or baseline.empty:
        todo = eligible.assign(reason="no_baseline")
        kept = 0
    else:
        joined = eligible.merge(
            baseline[KEY + ["anchor_found", LONGEST_HORIZON_COLUMN]], on=KEY, how="left", indicator=True
        )
        unpriced = joined["_merge"] == "left_only"
        open_window = joined["date"] >= now - pd.Timedelta(days=lookforward_days)
        not_complete = (joined["anchor_found"] != True) | joined[LONGEST_HORIZON_COLUMN].isna()  # noqa: E712
        incomplete = ~unpriced & open_window & not_complete.fillna(True)
        todo = joined.assign(reason=unpriced.map({True: "unpriced", False: "incomplete"}))
        todo = todo[unpriced | incomplete]
        kept = int((~(unpriced | incomplete)).sum())

    rows: dict[str, list] = {}
    for ticker, date, time in todo[KEY].itertuples(index=False):
        rows.setdefault(ticker, []).append((date.date(), None if pd.isna(time) else time))
    return PricePlan(rows=rows, reasons=Counter(todo["reason"]), kept=kept)


def merge_prices(baseline: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Baseline price rows with fresh ones winning on KEY."""
    parts = [p for p in (baseline, fresh) if p is not None and not p.empty]
    if not parts:
        return conform(pd.DataFrame(), PRICE_SCHEMA)
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="last")
    return conform(merged, PRICE_SCHEMA)


def join(
    earnings: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    price_from: datetime.date,
    price_to: datetime.date | None = None,
) -> pd.DataFrame:
    """Earnings rows dated price_from..price_to, left-joined to prices. Newest first."""
    rows = earnings[earnings["date"] >= pd.Timestamp(price_from)]
    if price_to:
        rows = rows[rows["date"] <= pd.Timestamp(price_to)]
    out = rows.merge(prices, on=KEY, how="left")
    out = out.sort_values(["date", "ticker"], ascending=[False, True], kind="stable")
    return conform(out, SCHEMA)
