"""Pins the parquet contract to the sister project's OUTPUT_COLUMNS."""

import dataclasses
import datetime

import pytest

from llm_engineering.application.api.fmp_models import (
    OUTPUT_COLUMNS,
    SNAPSHOT_COLUMNS,
    EarningsRecord,
)

# Copied verbatim from sip_news_data_platform/services/earnings_service.py.
SISTER_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "time",
    "timing_label",
    "date_corrected",
    "fiscal_date_ending",
    "eps_actual",
    "eps_estimated",
    "eps_surprise",
    "eps_surprise_pct",
    "revenue_actual",
    "revenue_estimated",
    "revenue_surprise",
    "revenue_surprise_pct",
    "sales_growth_yoy_pct",
]


def _record(**overrides) -> EarningsRecord:
    base = dict(
        ticker="AAPL",
        date=datetime.date(2026, 7, 30),
        time="amc",
        timing_label="After market close",
        date_corrected=False,
        fiscal_date_ending=datetime.date(2026, 6, 27),
        eps_actual=1.57,
        eps_estimated=1.43,
        revenue_actual=94_036_000_000.0,
        revenue_estimated=89_300_000_000.0,
    )
    return EarningsRecord(**{**base, **overrides})


def test_columns_match_the_sister_project_as_a_set() -> None:
    assert set(OUTPUT_COLUMNS) == set(SISTER_OUTPUT_COLUMNS)


def test_identity_columns_lead_and_derived_columns_trail() -> None:
    """Identity first, derived last; a reorder must be deliberate."""
    assert OUTPUT_COLUMNS[:2] == ("ticker", "date")
    assert OUTPUT_COLUMNS[-5:] == (
        "eps_surprise",
        "eps_surprise_pct",
        "revenue_surprise",
        "revenue_surprise_pct",
        "sales_growth_yoy_pct",
    )


def test_snapshot_adds_only_ingested_at() -> None:
    assert SNAPSHOT_COLUMNS == OUTPUT_COLUMNS + ("ingested_at",)


def test_derived_fields_default_to_none() -> None:
    record = _record()

    assert record.eps_surprise is None
    assert record.sales_growth_yoy_pct is None


def test_record_is_frozen() -> None:
    record = _record()

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.eps_actual = 2.0  # type: ignore[misc]


def test_replace_produces_a_new_record_with_metrics() -> None:
    record = _record()

    derived = dataclasses.replace(record, eps_surprise=0.14)

    assert derived.eps_surprise == 0.14
    assert record.eps_surprise is None
    assert derived.ticker == record.ticker
