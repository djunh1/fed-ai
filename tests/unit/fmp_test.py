"""FmpClient: key handling, retry opt-in, error redaction. No network."""

import dataclasses
import datetime
import json
import logging

import httpx
import pytest

from llm_engineering.application.api import fmp
from llm_engineering.application.api.fmp import (
    RETRY_STATUSES,
    FmpClient,
    FmpError,
    apply_authoritative_dates,
    dedupe,
    parse_earnings,
    parse_stable_dates,
    parse_universe,
    pct_change,
    reconcile_dates,
    sales_growth_yoy,
    with_derived_metrics,
)

URL = f"{fmp.V3_BASE}/stock/list"
KEY = "sekrit-key"


def _response(status: int, content: bytes = b"[]") -> httpx.Response:
    return httpx.Response(status, content=content, request=httpx.Request("GET", URL))


class FakeGet:
    """Stands in for http.get; records kwargs, returns or raises the queued result."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> bytes:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def fake_get(monkeypatch: pytest.MonkeyPatch):
    def install(result) -> FakeGet:
        fake = FakeGet(result)
        monkeypatch.setattr(fmp, "get", fake)
        return fake

    return install


def _status_error(status: int) -> httpx.HTTPStatusError:
    response = _response(status)
    return httpx.HTTPStatusError("boom", request=response.request, response=response)


# --- construction -----------------------------------------------------------


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fmp.settings, "FMP_API_KEY", None)

    with pytest.raises(RuntimeError, match="FMP_API_KEY"):
        FmpClient()


def test_key_falls_back_to_settings(monkeypatch: pytest.MonkeyPatch, fake_get) -> None:
    monkeypatch.setattr(fmp.settings, "FMP_API_KEY", KEY)
    fake = fake_get(b"[]")

    FmpClient()._get(URL)

    assert fake.calls[0]["params"]["apikey"] == KEY


def test_httpx_request_logging_is_silenced() -> None:
    logging.getLogger("httpx").setLevel(logging.INFO)

    FmpClient(KEY)

    assert logging.getLogger("httpx").level == logging.WARNING


# --- _get request shape -----------------------------------------------------


def test_apikey_added_and_none_params_dropped(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY)._get(URL, symbol="AAPL", limit=None)

    assert fake.calls[0]["params"] == {"symbol": "AAPL", "apikey": KEY}


def test_401_is_opted_into_retries(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY)._get(URL)

    assert fake.calls[0]["retry_statuses"] == RETRY_STATUSES
    assert 401 in RETRY_STATUSES


def test_no_cache_dir_is_passed(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY)._get(URL)

    assert "cache_dir" not in fake.calls[0]


def test_one_limiter_shared_across_calls(fake_get) -> None:
    fake = fake_get(b"[]")
    client = FmpClient(KEY)

    client._get(URL)
    client._get(URL)

    assert fake.calls[0]["limiter"] is fake.calls[1]["limiter"] is client._limiter


def test_decodes_json(fake_get) -> None:
    fake_get(json.dumps([{"symbol": "AAPL"}]).encode())

    assert FmpClient(KEY)._get(URL) == [{"symbol": "AAPL"}]


# --- errors -----------------------------------------------------------------


def test_http_error_becomes_fmp_error_with_status(fake_get) -> None:
    fake_get(_status_error(402))

    with pytest.raises(FmpError) as excinfo:
        FmpClient(KEY)._get(URL)

    assert excinfo.value.status == 402
    assert excinfo.value.path == URL


def test_error_message_never_contains_the_key(fake_get) -> None:
    fake_get(_status_error(401))

    with pytest.raises(FmpError) as excinfo:
        FmpClient(KEY)._get(f"{URL}?apikey={KEY}")

    assert KEY not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_transport_error_becomes_fmp_error(fake_get) -> None:
    fake_get(httpx.ConnectError("dns"))

    with pytest.raises(FmpError, match="ConnectError"):
        FmpClient(KEY)._get(URL)


def test_non_json_body_becomes_fmp_error(fake_get) -> None:
    fake_get(b"<html>maintenance</html>")

    with pytest.raises(FmpError, match="not JSON"):
        FmpClient(KEY)._get(URL)


def test_200_with_error_message_body_raises(fake_get) -> None:
    fake_get(json.dumps({"Error Message": "Limit Reach"}).encode())

    with pytest.raises(FmpError, match="Limit Reach"):
        FmpClient(KEY)._get(URL)


# --- helpers ----------------------------------------------------------------


@pytest.mark.parametrize("symbol,expected", [("AAPL", "AAPL"), ("BRK-B", "BRK-B"), ("ABC/WS", "ABC%2FWS"), ("BF.B", "BF.B")])
def test_symbol_path_quotes_slashes(symbol: str, expected: str) -> None:
    assert fmp._symbol_path(symbol) == expected


# --- screener + parse_universe -----------------------------------------------


def _row(symbol: str, name: str = "Acme Corp", cap: float = 1e9) -> dict:
    return {"symbol": symbol, "companyName": name, "marketCap": cap, "exchangeShortName": "NYSE"}


def test_screener_sends_the_sister_filters(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY).screener()

    call = fake.calls[0]
    assert call["url"] == f"{fmp.V3_BASE}/stock-screener"
    assert call["params"] == {
        "exchange": "NYSE,NASDAQ,AMEX",
        "priceMoreThan": 5.0,
        "marketCapMoreThan": 300_000_000,
        "volumeMoreThan": 200_000,
        "isActivelyTrading": "true",
        "isEtf": "false",
        "isFund": "false",
        "limit": fmp.SCREENER_LIMIT,
        "apikey": KEY,
    }


def test_screener_filters_are_overridable(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY).screener(min_price=1, min_market_cap=50e6, min_volume=10_000, exchanges=("NASDAQ",))

    params = fake.calls[0]["params"]
    assert (params["exchange"], params["priceMoreThan"], params["marketCapMoreThan"], params["volumeMoreThan"]) == (
        "NASDAQ", 1, 50_000_000, 10_000
    )


def test_screener_raises_when_truncated(fake_get) -> None:
    fake_get(json.dumps([_row(str(i)) for i in range(fmp.SCREENER_LIMIT)]).encode())

    with pytest.raises(FmpError, match="truncated"):
        FmpClient(KEY).screener()


@pytest.mark.parametrize(
    "name",
    [
        "Acme Units",
        "Foo Warrant",
        "Bar 5% Notes due 2030",
        "Baz Rights",
        "Qux American Depositary Shares",
        "Corp 8% Series H Preferred Stock",
        "Corp Subordinated Debentures",
        "Blank Check Holdings",
        "Zeta Acquisition Corp II",
        "Zeta Acquisition Company",
    ],
)
def test_universe_drops_non_common_and_shell_names(name: str) -> None:
    assert parse_universe([_row("X", name=name)]) == []


@pytest.mark.parametrize("name", ["UnitedHealth Group", "Curtiss-Wright Corp", "Rightmove plc", "Unity Software"])
def test_universe_keeps_word_boundary_false_positives(name: str) -> None:
    assert parse_universe([_row("X", name=name)]) == ["X"]


def test_universe_sorted_by_market_cap_desc_and_unique() -> None:
    records = [_row("B", cap=2e9), _row("A", cap=5e9), _row("B", cap=2e9), _row("C", cap=None)]

    assert parse_universe(records) == ["A", "B", "C"]


def test_universe_tolerates_missing_name_and_symbol() -> None:
    records = [_row("A", name=None), {"companyName": "No Symbol Inc"}]

    assert parse_universe(records) == ["A"]


def test_universe_stats_count_each_stage() -> None:
    records = [_row("A"), _row("B", name="B Warrants"), _row("C", name="C Acquisition Corp"), _row("A")]
    stats: dict = {}

    parse_universe(records, stats=stats)

    assert stats == {"screened": 4, "common": 3, "operating": 2, "symbols": 1}


def test_stock_list_hits_v3(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY).stock_list()

    assert fake.calls[0]["url"] == f"{fmp.V3_BASE}/stock/list"


# --- earnings_v3 + parse_earnings ---------------------------------------------


def _v3(**overrides) -> dict:
    row = {
        "date": "2026-07-30",
        "symbol": "AAPL",
        "eps": 1.57,
        "epsEstimated": 1.43,
        "time": "amc",
        "revenue": 94036000000,
        "revenueEstimated": 89300000000,
        "updatedFromDate": "2026-08-01",
        "fiscalDateEnding": "2026-06-27",
    }
    return {**row, **overrides}


def test_earnings_v3_url_quotes_symbol(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY).earnings_v3("ABC/WS")

    assert fake.calls[0]["url"] == f"{fmp.V3_BASE}/historical/earning_calendar/ABC%2FWS"


def test_parse_maps_every_field() -> None:
    [record] = parse_earnings("AAPL", [_v3()])

    assert record.ticker == "AAPL"
    assert record.date == datetime.date(2026, 7, 30)
    assert record.time == "amc"
    assert record.timing_label == "After market close"
    assert record.date_corrected is False
    assert record.fiscal_date_ending == datetime.date(2026, 6, 27)
    assert (record.eps_actual, record.eps_estimated) == (1.57, 1.43)
    assert (record.revenue_actual, record.revenue_estimated) == (94036000000.0, 89300000000.0)


def test_parse_revenue_ints_become_floats() -> None:
    [record] = parse_earnings("AAPL", [_v3()])

    assert isinstance(record.revenue_actual, float)


def test_parse_future_row_keeps_null_actuals() -> None:
    [record] = parse_earnings("AAPL", [_v3(eps=None, revenue=None)])

    assert record.eps_actual is None
    assert record.revenue_actual is None
    assert record.eps_estimated == 1.43


def test_parse_tolerates_missing_keys() -> None:
    [record] = parse_earnings("AAPL", [{"date": "1985-09-30"}])

    assert record.time is None
    assert record.fiscal_date_ending is None
    assert record.eps_actual is None


@pytest.mark.parametrize("time,label", [("bmo", "Before market open"), ("amc", "After market close"), ("dmh", "Unspecified"), (None, "Unspecified"), ("", "Unspecified")])
def test_parse_timing_labels(time, label: str) -> None:
    [record] = parse_earnings("AAPL", [_v3(time=time)])

    assert record.timing_label == label


def test_parse_empty_time_stored_as_none() -> None:
    [record] = parse_earnings("AAPL", [_v3(time="")])

    assert record.time is None


@pytest.mark.parametrize("bad", [None, "", "not-a-date", "2026-13-45"])
def test_parse_skips_rows_with_bad_date(bad) -> None:
    assert parse_earnings("AAPL", [_v3(date=bad), _v3()]) == parse_earnings("AAPL", [_v3()])


def test_parse_bad_fiscal_date_becomes_none() -> None:
    [record] = parse_earnings("AAPL", [_v3(fiscalDateEnding="garbage")])

    assert record.fiscal_date_ending is None


@pytest.mark.parametrize("raw,expected", [("1.5", 1.5), (2, 2.0), ("n/a", None), (True, None), ([], None)])
def test_parse_numeric_coercion(raw, expected) -> None:
    [record] = parse_earnings("AAPL", [_v3(eps=raw)])

    assert record.eps_actual == expected


def test_parse_ticker_comes_from_argument_not_row() -> None:
    [record] = parse_earnings("BRK-B", [_v3(symbol="BRK.B")])

    assert record.ticker == "BRK-B"


def test_parse_preserves_input_order() -> None:
    rows = [_v3(date="2026-07-30"), _v3(date="2026-04-30"), _v3(date="2026-01-29")]

    assert [r.date.month for r in parse_earnings("AAPL", rows)] == [7, 4, 1]


# --- stable endpoint + reconciliation ---------------------------------------


def d(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


def test_earnings_stable_request(fake_get) -> None:
    fake = fake_get(b"[]")

    FmpClient(KEY).earnings_stable("BRK-B")

    call = fake.calls[0]
    assert call["url"] == f"{fmp.STABLE_BASE}/earnings"
    assert call["params"] == {"symbol": "BRK-B", "limit": fmp.STABLE_HISTORY_LIMIT, "apikey": KEY}


def test_parse_stable_dates_sorted_and_skips_junk() -> None:
    rows = [{"date": "2026-07-30"}, {"date": None}, {"date": "bad"}, "not-a-dict", {"date": "2026-01-29"}]

    assert parse_stable_dates(rows) == [d("2026-01-29"), d("2026-07-30")]


def test_reconcile_exact_matches() -> None:
    reported = [d("2026-01-29"), d("2026-04-30")]

    assert reconcile_dates(reported, reported) == reported


def test_reconcile_one_day_shift() -> None:
    assert reconcile_dates([d("2026-07-29")], [d("2026-07-30")]) == [d("2026-07-30")]


def test_reconcile_beyond_tolerance_is_none() -> None:
    assert reconcile_dates([d("2026-07-01")], [d("2026-07-09")]) == [None]


def test_reconcile_at_tolerance_still_matches() -> None:
    assert reconcile_dates([d("2026-07-01")], [d("2026-07-08")]) == [d("2026-07-08")]


def test_reconcile_is_one_to_one() -> None:
    """Two reported dates near one authoritative date: only the nearer claims it."""
    result = reconcile_dates([d("2026-07-29"), d("2026-07-31")], [d("2026-07-30")])

    assert result == [d("2026-07-30"), None]


def test_reconcile_quarter_missing_from_authoritative_does_not_cascade() -> None:
    reported = [d("2026-01-28"), d("2026-04-29"), d("2026-07-29")]
    authoritative = [d("2026-01-29"), d("2026-07-30")]

    assert reconcile_dates(reported, authoritative) == [d("2026-01-29"), None, d("2026-07-30")]


def test_reconcile_extra_authoritative_quarter_is_ignored() -> None:
    reported = [d("2026-01-29"), d("2026-07-30")]
    authoritative = [d("2026-01-29"), d("2026-04-30"), d("2026-07-30")]

    assert reconcile_dates(reported, authoritative) == reported


def test_reconcile_preserves_input_order() -> None:
    """Input newest first, as v3 returns it; output aligned to it."""
    reported = [d("2026-07-29"), d("2026-04-29"), d("2026-01-28")]
    authoritative = [d("2026-01-29"), d("2026-04-30"), d("2026-07-30")]

    assert reconcile_dates(reported, authoritative) == [d("2026-07-30"), d("2026-04-30"), d("2026-01-29")]


def test_reconcile_empty_authoritative() -> None:
    assert reconcile_dates([d("2026-07-29")], []) == [None]


def test_apply_moves_and_flags_only_shifted_rows() -> None:
    records = parse_earnings("MSFT", [_v3(date="2026-07-29"), _v3(date="2026-04-30")])

    out = apply_authoritative_dates(records, [d("2026-04-30"), d("2026-07-30")])

    assert [(r.date, r.date_corrected) for r in out] == [(d("2026-07-30"), True), (d("2026-04-30"), False)]


def test_apply_unmatched_keeps_v3_date_unflagged() -> None:
    records = parse_earnings("MSFT", [_v3(date="2026-07-01")])

    [out] = apply_authoritative_dates(records, [d("2026-09-01")])

    assert (out.date, out.date_corrected) == (d("2026-07-01"), False)


def test_apply_empty_authoritative_is_a_no_op() -> None:
    records = parse_earnings("MSFT", [_v3()])

    assert apply_authoritative_dates(records, []) == records


def test_apply_leaves_other_fields_alone() -> None:
    [record] = parse_earnings("MSFT", [_v3(date="2026-07-29")])

    [out] = apply_authoritative_dates([record], [d("2026-07-30")])

    assert (out.eps_actual, out.time, out.fiscal_date_ending) == (record.eps_actual, record.time, record.fiscal_date_ending)


# --- derived metrics + dedupe -------------------------------------------------


@pytest.mark.parametrize(
    "actual,baseline,expected",
    [
        (1.57, 1.43, 9.79),
        (-0.20, -0.50, 60.0),  # beat a negative estimate: positive
        (-0.50, -0.20, -150.0),
        (1.0, 0, None),
        (1.0, None, None),
        (None, 1.0, None),
    ],
)
def test_pct_change(actual, baseline, expected) -> None:
    assert pct_change(actual, baseline) == expected


def _quarter(fde: str, revenue: float | None, date: str | None = None) -> fmp.EarningsRecord:
    [record] = parse_earnings("X", [_v3(date=date or fde, fiscalDateEnding=fde, revenue=revenue)])
    return record


def test_yoy_matches_prior_year_period() -> None:
    records = [_quarter("2026-06-27", 110.0), _quarter("2025-06-28", 100.0)]

    assert sales_growth_yoy(records) == [10.0, None]


def test_yoy_tolerates_53_week_year() -> None:
    """2026-07-04 minus 365 days is 2025-07-04; prior period 2025-06-28 is 6 days off."""
    records = [_quarter("2026-07-04", 120.0), _quarter("2025-06-28", 100.0)]

    assert sales_growth_yoy(records)[0] == 20.0


def test_yoy_none_beyond_tolerance() -> None:
    records = [_quarter("2026-06-27", 110.0), _quarter("2025-05-01", 100.0)]

    assert sales_growth_yoy(records)[0] is None


def test_yoy_none_when_prior_revenue_zero_or_missing() -> None:
    zero = [_quarter("2026-06-27", 110.0), _quarter("2025-06-28", 0.0)]
    missing = [_quarter("2026-06-27", 110.0), _quarter("2025-06-28", None)]

    assert sales_growth_yoy(zero)[0] is None
    assert sales_growth_yoy(missing)[0] is None


def test_yoy_duplicate_period_first_wins() -> None:
    records = [_quarter("2026-06-27", 110.0), _quarter("2025-06-28", 100.0), _quarter("2025-06-28", 50.0, date="2025-07-31")]

    assert sales_growth_yoy(records)[0] == 10.0


def test_yoy_row_without_period_does_not_break_neighbours() -> None:
    records = [_quarter("2026-06-27", 110.0), _quarter(None, 999.0, date="2025-12-01"), _quarter("2025-06-28", 100.0)]

    assert sales_growth_yoy(records) == [10.0, None, None]


def test_yoy_picks_nearest_of_several_candidates() -> None:
    records = [_quarter("2026-06-30", 110.0), _quarter("2025-06-01", 50.0), _quarter("2025-06-28", 100.0)]

    assert sales_growth_yoy(records)[0] == 10.0


def test_with_derived_metrics_fills_all_five() -> None:
    records = [_quarter("2026-06-27", 110.0), _quarter("2025-06-28", 100.0)]
    records = [dataclasses.replace(r, eps_actual=1.57, eps_estimated=1.43, revenue_estimated=100.0) for r in records]

    first = with_derived_metrics(records)[0]

    assert first.eps_surprise == pytest.approx(0.14)
    assert first.eps_surprise_pct == 9.79
    assert first.revenue_surprise == 10.0
    assert first.revenue_surprise_pct == 10.0
    assert first.sales_growth_yoy_pct == 10.0


def test_with_derived_metrics_future_row_has_no_surprise() -> None:
    [record] = parse_earnings("X", [_v3(eps=None, revenue=None)])

    [out] = with_derived_metrics([record])

    assert (out.eps_surprise, out.eps_surprise_pct, out.revenue_surprise, out.sales_growth_yoy_pct) == (None, None, None, None)


def test_dedupe_sorts_newest_first_and_keeps_first_seen() -> None:
    a = _quarter("2026-03-28", 1.0, date="2026-04-30")
    b = _quarter("2026-06-27", 2.0, date="2026-07-30")
    dup = _quarter("2026-03-28", 9.0, date="2026-04-30")

    assert dedupe([a, b, dup]) == [b, a]
