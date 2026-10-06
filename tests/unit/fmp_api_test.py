"""FmpEarningsApi with a fake client. No network."""

import datetime

import pytest

from llm_engineering.application.api import fmp
from llm_engineering.application.api.fmp import FmpError
from llm_engineering.application.api.fmp_api import FmpEarningsApi


def _v3(date: str, fde: str, revenue: float = 100.0, time: str = "amc") -> dict:
    return {"date": date, "eps": 1.5, "epsEstimated": 1.0, "time": time, "revenue": revenue, "revenueEstimated": 90.0, "fiscalDateEnding": fde}


class FakeClient:
    def __init__(self, *, v3=None, stable=None, screener=None, stable_error: Exception | None = None) -> None:
        self.v3 = v3 or {}
        self.stable = stable or {}
        self.screener_rows = screener or []
        self.stable_error = stable_error
        self.calls: list[tuple] = []

    def screener(self, **filters) -> list[dict]:
        self.calls.append(("screener", filters))
        return self.screener_rows

    def earnings_v3(self, symbol: str) -> list[dict]:
        self.calls.append(("v3", symbol))
        return self.v3.get(symbol, [])

    def earnings_stable(self, symbol: str) -> list[dict]:
        self.calls.append(("stable", symbol))
        if self.stable_error:
            raise self.stable_error
        return self.stable.get(symbol, [])


# --- construction -------------------------------------------------------------


def test_construction_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fmp.settings, "FMP_API_KEY", None)

    FmpEarningsApi()


def test_client_built_lazily_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fmp.settings, "FMP_API_KEY", "k")
    api = FmpEarningsApi()

    assert api.client is api.client


# --- universe -----------------------------------------------------------------


def test_universe_passes_filters_and_parses() -> None:
    rows = [
        {"symbol": "SMALL", "companyName": "Small Co", "marketCap": 1e9},
        {"symbol": "BIG", "companyName": "Big Co", "marketCap": 9e9},
        {"symbol": "W", "companyName": "W Warrants", "marketCap": 5e9},
    ]
    client = FakeClient(screener=rows)
    stats: dict = {}

    symbols = FmpEarningsApi(client).universe(min_price=1, exchanges=("NASDAQ",), stats=stats)

    assert symbols == ["BIG", "SMALL"]
    assert client.calls[0] == (
        "screener",
        {"min_price": 1, "min_market_cap": fmp.DEFAULT_MIN_MARKET_CAP, "min_volume": fmp.DEFAULT_MIN_VOLUME, "exchanges": ("NASDAQ",)},
    )
    assert stats["symbols"] == 2


# --- fetch_earnings -------------------------------------------------------------


def test_fetch_chains_reconcile_derive_and_sort() -> None:
    client = FakeClient(
        v3={"MSFT": [_v3("2025-07-29", "2025-06-30", 100.0), _v3("2026-07-29", "2026-06-30", 120.0)]},
        stable={"MSFT": [{"date": "2025-07-30"}, {"date": "2026-07-30"}]},
    )

    records = FmpEarningsApi(client).fetch_earnings("MSFT")

    assert [r.date for r in records] == [datetime.date(2026, 7, 30), datetime.date(2025, 7, 30)]
    assert all(r.date_corrected for r in records)
    assert records[0].sales_growth_yoy_pct == 20.0
    assert records[0].eps_surprise_pct == 50.0


def test_fetch_empty_v3_skips_stable() -> None:
    client = FakeClient()

    assert FmpEarningsApi(client).fetch_earnings("NONE") == []
    assert client.calls == [("v3", "NONE")]


def test_fetch_stable_failure_propagates() -> None:
    client = FakeClient(v3={"X": [_v3("2026-07-29", "2026-06-30")]}, stable_error=FmpError("/stable/earnings", status=503))

    with pytest.raises(FmpError):
        FmpEarningsApi(client).fetch_earnings("X")


def test_fetch_v3_failure_propagates_without_stable_call() -> None:
    class Failing(FakeClient):
        def earnings_v3(self, symbol):
            self.calls.append(("v3", symbol))
            raise FmpError("/v3", status=401)

    client = Failing()

    with pytest.raises(FmpError):
        FmpEarningsApi(client).fetch_earnings("X")
    assert client.calls == [("v3", "X")]


def test_fetch_empty_stable_keeps_v3_dates() -> None:
    client = FakeClient(v3={"X": [_v3("2026-07-29", "2026-06-30")]}, stable={"X": []})

    [record] = FmpEarningsApi(client).fetch_earnings("X")

    assert (record.date, record.date_corrected) == (datetime.date(2026, 7, 29), False)


def test_fetch_ticker_is_the_requested_symbol() -> None:
    client = FakeClient(v3={"BRK-B": [_v3("2026-07-29", "2026-06-30")]})

    [record] = FmpEarningsApi(client).fetch_earnings("BRK-B")

    assert record.ticker == "BRK-B"
