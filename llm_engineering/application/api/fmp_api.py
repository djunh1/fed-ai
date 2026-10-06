"""FMP earnings as one source: universe in, records per symbol out. No logic
of its own; fmp.py holds it. Not a BaseApi: no URLs, no Mongo (ADR 0001)."""

from llm_engineering.application.api import fmp
from llm_engineering.application.api.fmp_models import EarningsRecord


class FmpEarningsApi:
    def __init__(self, client: fmp.FmpClient | None = None) -> None:
        """Client built on first use, so constructing needs no key. Tests inject a fake."""
        self._client = client

    @property
    def client(self) -> fmp.FmpClient:
        if self._client is None:
            self._client = fmp.FmpClient()
        return self._client

    def universe(
        self,
        *,
        min_price: float = fmp.DEFAULT_MIN_PRICE,
        min_market_cap: float = fmp.DEFAULT_MIN_MARKET_CAP,
        min_volume: int = fmp.DEFAULT_MIN_VOLUME,
        exchanges: tuple[str, ...] = fmp.US_EXCHANGES,
        stats: dict | None = None,
    ) -> list[str]:
        """Screened symbols, largest market cap first. One call."""
        rows = self.client.screener(
            min_price=min_price, min_market_cap=min_market_cap, min_volume=min_volume, exchanges=exchanges
        )
        return fmp.parse_universe(rows, stats=stats)

    def fetch_earnings(self, symbol: str) -> list[EarningsRecord]:
        """Full reconciled history with derived metrics, newest first. Two calls.

        Empty v3 skips the stable call. A stable failure raises (ADR 0003): a
        failed symbol retries next run, a wrongly dated one would freeze.
        """
        records = fmp.parse_earnings(symbol, self.client.earnings_v3(symbol))
        if not records:
            return []
        authoritative = fmp.parse_stable_dates(self.client.earnings_stable(symbol))
        records = fmp.apply_authoritative_dates(records, authoritative)
        return fmp.dedupe(fmp.with_derived_metrics(records))
