"""One earnings announcement, keyed by (ticker, date). Same columns as the
sister project's OUTPUT_COLUMNS. Frozen dataclass, not pydantic: records are
in-memory only and become a frame at the snapshot boundary."""

import datetime
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class EarningsRecord:
    ticker: str
    date: datetime.date  # reconciled (ADR 0003)

    time: str | None  # "bmo", "amc", sometimes "dmh" or missing
    timing_label: str  # "Unspecified" unless bmo/amc
    date_corrected: bool
    fiscal_date_ending: datetime.date | None
    eps_actual: float | None  # None until reported
    eps_estimated: float | None
    revenue_actual: float | None
    revenue_estimated: float | None

    # Derived (ADR 0007); set later via dataclasses.replace().
    eps_surprise: float | None = None
    eps_surprise_pct: float | None = None
    revenue_surprise: float | None = None
    revenue_surprise_pct: float | None = None
    sales_growth_yoy_pct: float | None = None


#: Parquet contract, derived from the class so they cannot drift.
OUTPUT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(EarningsRecord))

#: ingested_at is stamped per run on the frame, not per record.
SNAPSHOT_COLUMNS: tuple[str, ...] = OUTPUT_COLUMNS + ("ingested_at",)
