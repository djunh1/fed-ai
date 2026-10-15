"""Enriched snapshot rules: what to price, merge, join."""

import datetime

import pandas as pd
import pytest

from llm_engineering.application import fmp_enriched as fe
from llm_engineering.application import fmp_snapshot
from llm_engineering.application.api.fmp_models import EarningsRecord
from llm_engineering.application.price_action import PRICE_ACTION_COLUMNS

TODAY = datetime.date(2026, 10, 6)
FROM = datetime.date(2016, 1, 1)
RUN = datetime.datetime(2026, 10, 6, 12, tzinfo=datetime.timezone.utc)
EARLIER = datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc)


def d(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


def earn(*rows: tuple[str, str, str | None]) -> pd.DataFrame:
    records = [
        EarningsRecord(
            ticker=t, date=d(day), time=time, timing_label="x", date_corrected=False, fiscal_date_ending=None,
            eps_actual=1.0, eps_estimated=1.0, revenue_actual=None, revenue_estimated=None,
        )
        for t, day, time in rows
    ]  # fmt: skip
    return fmp_snapshot.records_to_frame(records, ingested_at=EARLIER)


def priced(ticker: str, day: str, time: str | None, *, complete: bool = True, anchor: bool = True, **extra) -> dict:
    row = dict.fromkeys(PRICE_ACTION_COLUMNS)
    row.update(ticker=ticker, date=d(day), time=time, anchor_found=anchor)
    if anchor:
        row.update(date_0d=d(day), open_0d=100.0)
    if complete and anchor:
        row["70_d_max_gain"] = 5.0
    row.update(extra)
    return row


def plan(earnings, baseline=None):
    return fe.select_rows_to_price(earnings, baseline, today=TODAY, price_from=FROM, lookforward_days=107)


# --- schema --------------------------------------------------------------------


def test_schema_is_earnings_then_price_then_timestamps() -> None:
    cols = list(fe.SCHEMA)

    assert cols[:15] == list(fmp_snapshot.SCHEMA)[:15]
    assert cols[15:52] == list(PRICE_ACTION_COLUMNS)
    assert cols[-2:] == ["ingested_at", "priced_at"]
    assert fe.SCHEMA["anchor_found"] == "boolean"


def test_price_rows_to_frame_empty() -> None:
    assert fe.price_rows_to_frame([], priced_at=RUN).empty


# --- select_rows_to_price ---------------------------------------------------------


def test_no_baseline_prices_rows_in_window_only() -> None:
    e = earn(("A", "2026-07-30", "amc"), ("A", "2015-12-31", "amc"), ("A", "2026-10-29", "amc"))

    p = plan(e)

    assert p.rows == {"A": [(d("2026-07-30"), "amc")]}
    assert p.reasons == {"no_baseline": 1}


def test_today_is_eligible() -> None:
    assert plan(earn(("A", "2026-10-06", "bmo"))).rows == {"A": [(d("2026-10-06"), "bmo")]}


def test_unpriced_key_is_priced() -> None:
    e = earn(("A", "2026-01-29", "amc"), ("B", "2026-01-28", "bmo"))
    base = fe.price_rows_to_frame([priced("A", "2026-01-29", "amc")], priced_at=EARLIER)

    p = plan(e, base)

    assert p.rows == {"B": [(d("2026-01-28"), "bmo")]}
    assert (p.reasons, p.kept) == ({"unpriced": 1}, 1)


def test_changed_timing_is_a_new_key() -> None:
    e = earn(("A", "2026-01-29", "bmo"))
    base = fe.price_rows_to_frame([priced("A", "2026-01-29", None)], priced_at=EARLIER)

    assert plan(e, base).reasons == {"unpriced": 1}


def test_null_time_matches_null_time() -> None:
    e = earn(("A", "2026-01-29", None))
    base = fe.price_rows_to_frame([priced("A", "2026-01-29", None)], priced_at=EARLIER)

    assert plan(e, base).kept == 1


@pytest.mark.parametrize(
    "day,complete,anchor,repriced",
    [
        ("2026-09-01", False, True, True),  # horizon still open
        ("2026-09-01", True, True, False),  # already complete
        ("2026-09-01", False, False, True),  # had no session when priced
        ("2026-06-21", False, True, True),  # 107 days back: still open
        ("2026-06-20", False, True, False),  # 108 days back: final (e.g. delisted)
    ],
)
def test_incomplete_rule(day: str, complete: bool, anchor: bool, repriced: bool) -> None:
    e = earn(("A", day, "amc"))
    base = fe.price_rows_to_frame([priced("A", day, "amc", complete=complete, anchor=anchor)], priced_at=EARLIER)

    assert plan(e, base).reasons == ({"incomplete": 1} if repriced else {})


def test_rows_grouped_by_ticker() -> None:
    e = earn(("A", "2026-07-30", "amc"), ("B", "2026-07-29", "bmo"), ("A", "2026-04-30", "amc"))

    rows = plan(e).rows

    assert sorted(rows["A"]) == [(d("2026-04-30"), "amc"), (d("2026-07-30"), "amc")]
    assert rows["B"] == [(d("2026-07-29"), "bmo")]


# --- merge + join -------------------------------------------------------------------


def test_merge_prices_fresh_wins() -> None:
    base = fe.price_rows_to_frame([priced("A", "2026-01-29", "amc", open_0d=1.0)], priced_at=EARLIER)
    fresh = fe.price_rows_to_frame([priced("A", "2026-01-29", "amc", open_0d=2.0)], priced_at=RUN)

    out = fe.merge_prices(base, fresh)

    assert out["open_0d"].tolist() == [2.0]
    assert out["priced_at"].tolist() == [pd.Timestamp(RUN)]


def test_merge_prices_all_empty() -> None:
    assert fe.merge_prices(None, fe.price_rows_to_frame([], priced_at=RUN)).empty


def test_join_takes_earnings_columns_from_earnings_and_filters_floor() -> None:
    e = earn(("A", "2026-01-29", "amc"), ("A", "2015-10-27", "amc"), ("B", "2026-10-29", "bmo"))
    prices = fe.price_rows_to_frame([priced("A", "2026-01-29", "amc", open_0d=50.0)], priced_at=RUN)

    out = fe.join(e, prices, price_from=FROM)

    assert list(out.columns) == list(fe.SCHEMA)
    assert out[["ticker", "date"]].astype(str).values.tolist() == [["B", "2026-10-29"], ["A", "2026-01-29"]]
    row = out[out["ticker"] == "A"].iloc[0]
    assert row["open_0d"] == 50.0 and row["eps_actual"] == 1.0
    assert pd.isna(out[out["ticker"] == "B"]["anchor_found"].iloc[0])


def test_join_drops_prices_for_keys_no_longer_in_earnings() -> None:
    e = earn(("A", "2026-01-30", "amc"))  # date was corrected from 01-29
    prices = fe.price_rows_to_frame([priced("A", "2026-01-29", "amc")], priced_at=RUN)

    out = fe.join(e, prices, price_from=FROM)

    assert len(out) == 1 and pd.isna(out["anchor_found"].iloc[0])


def test_prices_from_enriched_round_trip() -> None:
    e = earn(("A", "2026-01-29", "amc"))
    prices = fe.price_rows_to_frame([priced("A", "2026-01-29", "amc")], priced_at=RUN)

    back = fe.prices_from_enriched(fe.join(e, prices, price_from=FROM))

    pd.testing.assert_frame_equal(back, prices)


# --- cutoff (price_to) ---------------------------------------------------------


def test_select_respects_price_to() -> None:
    e = earn(("A", "2026-01-29", "amc"), ("A", "2025-10-30", "amc"), ("B", "2026-07-30", "bmo"))

    p = fe.select_rows_to_price(e, None, today=TODAY, price_from=FROM, lookforward_days=107, price_to=d("2026-01-29"))

    assert p.rows == {"A": [(d("2026-01-29"), "amc"), (d("2025-10-30"), "amc")]}  # cutoff inclusive


def test_select_price_to_after_today_still_stops_at_today() -> None:
    e = earn(("A", "2026-10-29", "amc"))  # future
    p = fe.select_rows_to_price(e, None, today=TODAY, price_from=FROM, lookforward_days=107, price_to=d("2030-01-01"))
    assert p.rows == {}


def test_join_drops_rows_after_price_to() -> None:
    e = earn(("A", "2026-01-29", "amc"), ("B", "2026-07-30", "bmo"))
    prices = fe.price_rows_to_frame([priced("B", "2026-07-30", "bmo")], priced_at=RUN)

    out = fe.join(e, prices, price_from=FROM, price_to=d("2026-06-30"))

    assert out["ticker"].tolist() == ["A"]
