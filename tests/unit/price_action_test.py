"""Price action math on hand-built bars. Values checkable by hand."""

import datetime

import pytest

from llm_engineering.application import price_action as pa
from llm_engineering.application.price_action import Bar


def d(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


def bars_from(start: str, closes: list[float]) -> list[Bar]:
    """One bar per weekday. open = previous close, high/low = close +/- 1."""
    out, day, prev = [], d(start), closes[0]
    for close in closes:
        while day.weekday() >= 5:
            day += datetime.timedelta(days=1)
        out.append(
            Bar(
                day,
                open=prev,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1000.0,
            )
        )
        prev, day = close, day + datetime.timedelta(days=1)
    return out


# --- columns ------------------------------------------------------------------


def test_columns_cover_sister_enriched_plus_70d() -> None:
    sister = {
        "date_0d", "anchor_found", "open_0d", "close_0d", "high_0d", "low_0d", "volume",
        "percent_gap_up", "c-pc_pct", "max_gain_0d", "max_down_intraday",
        "1_d", "2_d", "3_d", "4_d", "5_d", "6_d", "7_d", "8_d", "9_d", "10_d", "20_d", "40_d", "60_d",
        "10_d_max_gain", "20_d_max_gain", "40_d_max_gain", "60_d_max_gain",
        "10_d_max_draw", "20_d_max_draw", "40_d_max_draw", "60_d_max_draw",
        "sector", "industry", "market_capitalization",
    }  # fmt: skip
    assert set(pa.PRICE_ACTION_COLUMNS) == sister | {"70_d_max_gain", "day_0_check"}


def test_lookforward_covers_70_sessions() -> None:
    assert pa.lookforward_days() == 107
    assert pa.LONGEST_HORIZON_COLUMN == "70_d_max_gain"


# --- parsing ------------------------------------------------------------------


def test_parse_bars_ascending_deduped_skips_bad() -> None:
    payload = {
        "historical": [
            {
                "date": "2026-07-31",
                "open": 2,
                "high": 3,
                "low": 1,
                "close": 2.5,
                "volume": 10,
            },
            {
                "date": "2026-07-30",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
            },
            {
                "date": "2026-07-30",
                "open": 9,
                "high": 9,
                "low": 9,
                "close": 9,
                "volume": 9,
            },
            {
                "date": "2026-07-29",
                "open": None,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            },
            {"date": "bad"},
        ]
    }
    bars = pa.parse_bars(payload)

    assert [b.date for b in bars] == [d("2026-07-30"), d("2026-07-31")]
    assert bars[0].open == 1.0


@pytest.mark.parametrize("payload", [None, [], {}, {"historical": None}, "x"])
def test_parse_bars_empty_shapes(payload) -> None:
    assert pa.parse_bars(payload) == []


def test_parse_market_caps() -> None:
    rows = [
        {"date": "2026-07-31", "marketCap": 2e12},
        {"date": "2026-07-30", "marketCap": 1e12},
        {"date": "x"},
    ]

    assert pa.parse_market_caps(rows) == [
        (d("2026-07-30"), 1e12),
        (d("2026-07-31"), 2e12),
    ]


def test_parse_profile() -> None:
    assert pa.parse_profile([{"sector": "Technology", "industry": ""}]) == {
        "sector": "Technology",
        "industry": None,
    }
    assert pa.parse_profile([]) == {"sector": None, "industry": None}


def test_market_cap_windows_cover_range_without_overlap() -> None:
    windows = pa.market_cap_windows(d("2016-01-01"), d("2020-12-31"))

    assert windows[0][0] == d("2016-01-01") and windows[-1][1] == d("2020-12-31")
    assert all(b[0] == a[1] + datetime.timedelta(days=1) for a, b in zip(windows, windows[1:]))
    assert all((e - s).days <= pa.MARKET_CAP_CHUNK_DAYS for s, e in windows)


def test_market_cap_as_of_falls_back_to_prior_day() -> None:
    caps = [(d("2026-07-29"), 1.0), (d("2026-07-31"), 3.0)]

    assert pa.market_cap_as_of(caps, d("2026-07-31")) == 3.0
    assert pa.market_cap_as_of(caps, d("2026-07-30")) == 1.0
    assert pa.market_cap_as_of(caps, d("2026-07-28")) is None
    assert pa.market_cap_as_of([], d("2026-07-30")) is None


# --- anchoring ----------------------------------------------------------------

WEEK = bars_from("2026-07-27", [10, 11, 12, 13, 14])  # Mon 27 .. Fri 31


@pytest.mark.parametrize(
    "announce,timing,expected",
    [
        ("2026-07-29", "bmo", "2026-07-29"),  # before open: same session
        ("2026-07-29", "BMO", "2026-07-29"),
        ("2026-07-29", "amc", "2026-07-30"),  # after close: next session
        ("2026-07-29", None, "2026-07-30"),  # unknown treated as amc
        ("2026-07-29", "dmh", "2026-07-30"),
        ("2026-07-26", "bmo", "2026-07-27"),  # Sunday rolls to Monday
        ("2026-07-26", "amc", "2026-07-27"),
    ],
)
def test_find_anchor(announce: str, timing, expected: str) -> None:
    i = pa.find_anchor_index(WEEK, d(announce), timing)

    assert WEEK[i].date == d(expected)


def test_find_anchor_long_weekend_is_within_cap() -> None:
    """Thu amc before Good Friday: day 0 is Monday, 4 days later."""
    bars = [Bar(d("2026-04-02"), 1, 1, 1, 1, 1), Bar(d("2026-04-06"), 1, 1, 1, 1, 1)]

    assert bars[pa.find_anchor_index(bars, d("2026-04-02"), "amc")].date == d("2026-04-06")


def test_find_anchor_none_when_stock_was_not_trading() -> None:
    """Pre-IPO filing: first session is months later, so no anchor."""
    bars = [Bar(d("2021-01-15"), 1, 1, 1, 1, 1)]

    assert pa.find_anchor_index(bars, d("2020-09-30"), "bmo") is None


@pytest.mark.parametrize("gap,found", [(5, True), (6, False)])
def test_find_anchor_cap_boundary(gap: int, found: bool) -> None:
    bars = [Bar(d("2026-07-01") + datetime.timedelta(days=gap), 1, 1, 1, 1, 1)]

    assert (pa.find_anchor_index(bars, d("2026-07-01"), "bmo") is not None) is found


def test_find_anchor_none_after_last_session() -> None:
    assert pa.find_anchor_index(WEEK, d("2026-07-31"), "amc") is None
    assert pa.find_anchor_index([], d("2026-07-31"), "bmo") is None


# --- metrics --------------------------------------------------------------------


def test_day0_metrics() -> None:
    bars = [
        Bar(d("2026-07-29"), 100, 101, 99, 100, 5),
        Bar(d("2026-07-30"), 110, 121, 104.5, 115.5, 7),  # gap up 10%
    ]
    m = pa.compute_row_metrics(bars, 1)

    assert m["date_0d"] == d("2026-07-30")
    assert (m["open_0d"], m["close_0d"], m["high_0d"], m["low_0d"], m["volume"]) == (
        110,
        115.5,
        121,
        104.5,
        7,
    )
    assert m["percent_gap_up"] == 10.0
    assert m["c-pc_pct"] == 15.5
    assert m["max_gain_0d"] == 10.0
    assert m["max_down_intraday"] == -5.0


def test_no_prior_bar_means_no_gap() -> None:
    m = pa.compute_row_metrics(WEEK, 0)

    assert "percent_gap_up" not in m and "c-pc_pct" not in m
    assert "max_gain_0d" in m


def test_zero_open_stops_after_ohlcv() -> None:
    m = pa.compute_row_metrics([Bar(d("2026-07-30"), 0, 1, 0, 1, 1)], 0)

    assert "max_gain_0d" not in m and m["open_0d"] == 0


def test_forward_return_uses_close_of_nth_session_vs_open() -> None:
    bars = bars_from("2026-01-05", [100.0] + [100.0 + i for i in range(1, 80)])
    m = pa.compute_row_metrics(bars, 0)  # open_0d = 100

    assert m["1_d"] == 1.0  # close of session 1 = 101
    assert m["10_d"] == 10.0
    assert m["60_d"] == 60.0


def test_max_gain_and_draw_span_day0_through_n() -> None:
    closes = [100.0] * 12
    closes[3], closes[7] = 120.0, 80.0
    m = pa.compute_row_metrics(bars_from("2026-01-05", closes), 0)

    assert m["10_d_max_gain"] == 21.0  # high 121
    assert m["10_d_max_draw"] == -21.0  # low 79


def test_horizons_past_the_data_are_absent() -> None:
    m = pa.compute_row_metrics(bars_from("2026-01-05", [100.0] * 15), 0)

    assert "10_d" in m and "20_d" not in m and "10_d_max_gain" in m and "20_d_max_gain" not in m


def test_rounding_two_places() -> None:
    bars = [
        Bar(d("2026-07-30"), 3.0, 3.0, 3.0, 3.0, 1),
        Bar(d("2026-07-31"), 3.0, 3.0, 3.0, 4.0, 1),
    ]

    assert pa.compute_row_metrics(bars, 0)["1_d"] == 33.33


# --- price_action_for -------------------------------------------------------------


def test_price_action_for_returns_every_column() -> None:
    out = pa.price_action_for(
        WEEK,
        [(d("2026-07-01"), 5e9)],
        {"sector": "Tech", "industry": "Chips"},
        d("2026-07-28"),
        "amc",
    )

    assert set(out) == set(pa.PRICE_ACTION_COLUMNS)
    assert out["anchor_found"] is True
    assert out["date_0d"] == d("2026-07-29")
    assert out["market_capitalization"] == 5e9
    assert (out["sector"], out["industry"]) == ("Tech", "Chips")
    assert out["60_d"] is None


def test_price_action_for_no_anchor_keeps_profile_only() -> None:
    out = pa.price_action_for(
        WEEK,
        [(d("2026-07-01"), 5e9)],
        {"sector": "Tech", "industry": None},
        d("2026-07-31"),
        "amc",
    )

    assert out["anchor_found"] is False
    assert out["date_0d"] is None and out["open_0d"] is None and out["market_capitalization"] is None
    assert out["sector"] == "Tech"


# --- day-0 check ---------------------------------------------------------------


def gapped(gaps: dict[int, float], n: int = 25) -> list[Bar]:
    """Flat 100 bars, range 1 (ATR 1); `gaps` opens bar i at 100 + gap."""
    out, day = [], d("2024-01-01")
    for i in range(n):
        while day.weekday() >= 5:
            day += datetime.timedelta(days=1)
        o = 100 + gaps.get(i, 0.0)
        out.append(
            Bar(
                day,
                open=o,
                high=max(o, 100) + 0.5,
                low=min(o, 100) - 0.5,
                close=100,
                volume=1.0,
            )
        )
        day += datetime.timedelta(days=1)
    return out


def test_day_0_check_ok_when_day_0_has_the_gap() -> None:
    assert pa.check_day_0(gapped({20: 5}), 20) == "ok"


def test_day_0_check_quiet_is_ok() -> None:
    assert pa.check_day_0(gapped({}), 20) == "ok"


def test_day_0_check_late_when_prior_session_gapped() -> None:
    assert pa.check_day_0(gapped({19: 5}), 20) == "late"


def test_day_0_check_early_when_next_session_gapped() -> None:
    assert pa.check_day_0(gapped({21: -5}), 20) == "early"


def test_day_0_check_needs_more_than_threshold_and_day_0() -> None:
    assert pa.check_day_0(gapped({21: 1.5}), 20) == "ok"  # under 2 ATR
    assert pa.check_day_0(gapped({20: 6, 21: 5}), 20) == "ok"  # day 0 bigger


def test_day_0_check_atr_ignores_day_minus_1_and_day_0() -> None:
    # A huge day -1 gap must not inflate the ATR and hide itself.
    assert pa.check_day_0(gapped({19: 50}), 20) == "late"


def test_day_0_check_none_without_history_or_next_session() -> None:
    bars = gapped({}, n=25)
    first = pa.ATR_SESSIONS + 2  # earliest anchor with a full ATR window
    assert pa.check_day_0(bars, first - 1) is None
    assert pa.check_day_0(bars, first) == "ok"
    assert pa.check_day_0(bars, 24) is None


def test_price_action_for_sets_day_0_check() -> None:
    bars = gapped({19: 5})
    row = pa.price_action_for(bars, [], {}, bars[19].date, "amc")  # amc: day 0 = bar 20
    assert row["day_0_check"] == "late"
