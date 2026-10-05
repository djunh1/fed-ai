"""The record the FMP earnings pipeline passes between layers.

One announcement, one record, keyed by (ticker, date). The field list is the
sister project's OUTPUT_COLUMNS in the same order, so the parquet this
pipeline writes is column-for-column what `sip_news_data_platform` already
produces for itself -- the consumer needs no mapping step.

Deliberately not pydantic and not a Mongo document: nothing here is stored
row-by-row. Records live in memory for one symbol at a time, then become a
pandas frame at the snapshot boundary (application/fmp_snapshot.py). A frozen
dataclass with slots is the cheapest thing that is still typed, hashable and
safe to pass through pure functions that return modified copies.

Kept separate from models.py on purpose: FedSourceDocument is typed to FOMC
documents (url, meeting_date, info_type), and this record shares none of that.
"""

import datetime
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class EarningsRecord:
    # --- identity -----------------------------------------------------
    ticker: str
    date: datetime.date  # announcement date, after reconciliation (ADR 0003)

    # --- straight from v3 historical/earning_calendar -----------------
    time: str | None  # "bmo", "amc", occasionally "dmh" or missing
    timing_label: str  # English for `time`; "Unspecified" when not bmo/amc
    date_corrected: bool  # True when the stable endpoint moved `date`
    fiscal_date_ending: datetime.date | None
    eps_actual: float | None  # None for a scheduled, not yet reported, quarter
    eps_estimated: float | None
    revenue_actual: float | None
    revenue_estimated: float | None

    # --- derived, filled by fmp.with_derived_metrics (ADR 0007) ---------
    # Defaulted so parse_earnings() can build a record before the metrics
    # exist; the derivation step returns a replace()d copy with them set.
    eps_surprise: float | None = None
    eps_surprise_pct: float | None = None
    revenue_surprise: float | None = None
    revenue_surprise_pct: float | None = None
    sales_growth_yoy_pct: float | None = None


#: The parquet contract, derived from the class so the two cannot drift.
#: Matches the sister project's OUTPUT_COLUMNS exactly, in order.
OUTPUT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(EarningsRecord))

#: `ingested_at` is a frame-level column, stamped once per run at the
#: snapshot boundary, so it is not a field on the record.
SNAPSHOT_COLUMNS: tuple[str, ...] = OUTPUT_COLUMNS + ("ingested_at",)
