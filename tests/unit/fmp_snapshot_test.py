"""Snapshot rules: schema, refresh selection, merge. Explicit `today` throughout."""

import datetime

import pandas as pd
import pytest

from llm_engineering.application.api.fmp_models import SNAPSHOT_COLUMNS, EarningsRecord
from llm_engineering.application.fmp_snapshot import (
    SCHEMA,
    empty_frame,
    merge_snapshot,
    records_to_frame,
    select_symbols_to_refresh,
)

TODAY = datetime.date(2026, 10, 6)
RUN = datetime.datetime(2026, 10, 6, 12, 0, tzinfo=datetime.timezone.utc)
EARLIER = datetime.datetime(2026, 9, 1, 12, 0, tzinfo=datetime.timezone.utc)


def rec(ticker: str, date: str, eps: float | None = 1.0) -> EarningsRecord:
    return EarningsRecord(
        ticker=ticker,
        date=datetime.date.fromisoformat(date),
        time="amc",
        timing_label="After market close",
        date_corrected=False,
        fiscal_date_ending=None,
        eps_actual=eps,
        eps_estimated=1.0,
        revenue_actual=None,
        revenue_estimated=None,
    )


def frame(*records: EarningsRecord, at: datetime.datetime = EARLIER) -> pd.DataFrame:
    return records_to_frame(records, ingested_at=at)


def plan(baseline, universe, stale_after_days: int = 75):
    return select_symbols_to_refresh(baseline, universe, today=TODAY, stale_after_days=stale_after_days)


# --- schema + frames ------------------------------------------------------------


def test_schema_covers_snapshot_columns_in_order() -> None:
    assert tuple(SCHEMA) == SNAPSHOT_COLUMNS


def test_empty_frame_has_schema() -> None:
    out = empty_frame()

    assert out.empty
    assert {c: str(t) for c, t in out.dtypes.items()} == SCHEMA


def test_records_to_frame_types_and_stamp() -> None:
    out = frame(rec("AAPL", "2026-07-30"), rec("AAPL", "2026-10-29", eps=None), at=RUN)

    assert {c: str(t) for c, t in out.dtypes.items()} == SCHEMA
    assert (out["ingested_at"] == pd.Timestamp(RUN)).all()
    assert pd.isna(out.loc[1, "eps_actual"])


def test_records_to_frame_empty() -> None:
    assert records_to_frame([], ingested_at=RUN).empty


# --- select_symbols_to_refresh --------------------------------------------------


def test_no_baseline_refreshes_everything_in_universe_order() -> None:
    result = plan(None, ["NVDA", "AAPL", "NVDA"])

    assert result.refresh == ["NVDA", "AAPL"]
    assert result.reasons == {"NVDA": "no_baseline", "AAPL": "no_baseline"}


def test_empty_baseline_is_no_baseline() -> None:
    assert plan(empty_frame(), ["A"]).reasons == {"A": "no_baseline"}


def test_new_symbol() -> None:
    result = plan(frame(rec("A", "2026-09-01")), ["A", "B"])

    assert result.refresh == ["B"]
    assert result.reasons == {"B": "new"}
    assert result.kept == ["A"]


@pytest.mark.parametrize("latest,stale", [("2026-07-23", False), ("2026-07-22", True)])
def test_stale_boundary(latest: str, stale: bool) -> None:
    """2026-07-23 is 75 days before TODAY: kept. One day earlier: stale."""
    result = plan(frame(rec("A", latest)), ["A"])

    assert (result.reasons.get("A") == "stale") is stale


def test_future_scheduled_row_keeps_symbol_fresh() -> None:
    """Old history plus a scheduled future date: latest is in the future, not stale."""
    result = plan(frame(rec("A", "2026-05-01"), rec("A", "2026-10-29", eps=None)), ["A"])

    assert result.kept == ["A"]


def test_past_null_inside_window_is_unresolved() -> None:
    result = plan(frame(rec("A", "2026-10-01", eps=None), rec("A", "2026-07-30")), ["A"])

    assert result.reasons == {"A": "unresolved"}


def test_future_null_is_not_unresolved() -> None:
    result = plan(frame(rec("A", "2026-10-06", eps=None), rec("A", "2026-09-01")), ["A"])

    assert result.kept == ["A"]


def test_old_null_outside_window_is_ignored() -> None:
    """TSLA 2008-style permanent gap: must not trigger a refetch every run."""
    result = plan(frame(rec("A", "2008-12-31", eps=None), rec("A", "2026-09-01")), ["A"])

    assert result.kept == ["A"]


def test_stale_wins_over_unresolved() -> None:
    result = plan(frame(rec("A", "2026-07-20", eps=None)), ["A"], stale_after_days=10)

    assert result.reasons == {"A": "stale"}


def test_dropped_symbols_reported_not_refreshed() -> None:
    result = plan(frame(rec("A", "2026-09-01"), rec("GONE", "2026-01-01")), ["A"])

    assert result.dropped == ["GONE"]
    assert "GONE" not in result.refresh


def test_refresh_keeps_universe_order() -> None:
    result = plan(frame(rec("KEEP", "2026-09-01")), ["Z", "KEEP", "A", "M"])

    assert result.refresh == ["Z", "A", "M"]


# --- merge_snapshot -------------------------------------------------------------


def test_merge_without_baseline_is_fresh() -> None:
    fresh = frame(rec("A", "2026-07-30"), at=RUN)

    out = merge_snapshot(None, fresh, succeeded={"A"})

    pd.testing.assert_frame_equal(out, fresh)


def test_merge_replaces_succeeded_symbols_wholesale() -> None:
    """A's old 2026-04-30 row is gone because A was refetched without it."""
    baseline = frame(rec("A", "2026-04-30"), rec("A", "2026-07-30"), rec("B", "2026-07-29"))
    fresh = frame(rec("A", "2026-07-30", eps=2.0), at=RUN)

    out = merge_snapshot(baseline, fresh, succeeded={"A"})

    assert out[["ticker", "date"]].astype(str).values.tolist() == [["A", "2026-07-30"], ["B", "2026-07-29"]]
    assert out.loc[out["ticker"] == "A", "eps_actual"].item() == 2.0


def test_merge_keeps_failed_symbols_baseline_rows() -> None:
    baseline = frame(rec("A", "2026-07-30"), rec("FAILED", "2026-07-29"))
    fresh = frame(rec("A", "2026-07-30"), at=RUN)

    out = merge_snapshot(baseline, fresh, succeeded={"A"})

    assert set(out["ticker"]) == {"A", "FAILED"}


def test_merge_preserves_ingested_at_of_kept_rows() -> None:
    baseline = frame(rec("A", "2026-07-30"), rec("B", "2026-07-29"), at=EARLIER)
    fresh = frame(rec("A", "2026-07-30"), at=RUN)

    out = merge_snapshot(baseline, fresh, succeeded={"A"}).set_index("ticker")["ingested_at"]

    assert out["A"] == pd.Timestamp(RUN)
    assert out["B"] == pd.Timestamp(EARLIER)


def test_merge_fresh_wins_on_duplicate_key() -> None:
    """Defensive: fresh rows for a symbol not in `succeeded` still win ties."""
    baseline = frame(rec("A", "2026-07-30", eps=1.0))
    fresh = frame(rec("A", "2026-07-30", eps=9.0), at=RUN)

    out = merge_snapshot(baseline, fresh, succeeded=set())

    assert out["eps_actual"].tolist() == [9.0]


def test_merge_sorts_newest_then_ticker() -> None:
    baseline = frame(rec("B", "2026-07-30"), rec("A", "2026-01-29"))
    fresh = frame(rec("A", "2026-07-30"), at=RUN)

    out = merge_snapshot(baseline, fresh, succeeded={"A"})

    assert out[["ticker", "date"]].astype(str).values.tolist() == [["A", "2026-07-30"], ["B", "2026-07-30"]]


def test_merge_all_empty() -> None:
    out = merge_snapshot(None, empty_frame(), succeeded=set())

    pd.testing.assert_frame_equal(out, empty_frame())


def test_merge_output_has_schema() -> None:
    out = merge_snapshot(frame(rec("B", "2026-07-30")), frame(rec("A", "2026-07-30"), at=RUN), succeeded={"A"})

    assert {c: str(t) for c, t in out.dtypes.items()} == SCHEMA
