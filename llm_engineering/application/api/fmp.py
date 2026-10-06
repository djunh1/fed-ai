"""Financial Modeling Prep client. Key travels as `apikey=` in the query, so
no HTTP cache and no raw httpx errors escape (ADR 0005)."""

import dataclasses
import datetime
import json
import logging
import re
from urllib.parse import quote

import httpx
from loguru import logger

from llm_engineering.application.api.fmp_models import EarningsRecord
from llm_engineering.infrastructure.http import RateLimiter, get
from llm_engineering.settings import settings

V3_BASE = "https://financialmodelingprep.com/api/v3"
STABLE_BASE = "https://financialmodelingprep.com/stable"

#: 401 is FMP's throttle signal as well as its bad-key reply.
RETRY_STATUSES = (401, 429)

US_EXCHANGES = ("NYSE", "NASDAQ", "AMEX")

#: Screener defaults, copied from the sister project (ADR 0006). $5 is the
#: SEC penny-stock line; $300M the small-cap floor; 200k shares the liquidity floor.
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_MARKET_CAP = 300_000_000
DEFAULT_MIN_VOLUME = 200_000

#: The screener truncates silently at `limit`; ask for far more than exists.
SCREENER_LIMIT = 25_000

#: Listings that do not report their own earnings. Word-bounded so
#: "UnitedHealth" and "Curtiss-Wright" survive.
NON_COMMON_PATTERN = re.compile(
    r"\b(?:warrants?|units?|rights?|preferred|depositary|debentures?|notes? due)\b",
    re.IGNORECASE,
)

#: Blank-check vehicles: no operating business, no meaningful earnings.
SHELL_PATTERN = re.compile(r"(?:acquisition corp|acquisition compan|blank check)", re.IGNORECASE)

TIMING_LABELS = {"bmo": "Before market open", "amc": "After market close"}

#: Quarters sit ~90 days apart; 7 days absorbs the 1-day v3 error safely.
DATE_MATCH_TOLERANCE_DAYS = 7

#: Above any symbol's history (~165 rows max).
STABLE_HISTORY_LIMIT = 1000

#: Prior-year period search window; absorbs 52/53-week fiscal calendars.
YOY_PERIOD_TOLERANCE_DAYS = 45


class FmpError(RuntimeError):
    """An FMP call failed. Message carries path and status only, never the key."""

    def __init__(self, path: str, *, status: int | None = None, reason: str = "") -> None:
        self.path = path
        self.status = status
        detail = f"HTTP {status}" if status is not None else reason
        super().__init__(f"FMP {path}: {detail}")


class FmpClient:
    """One limiter per instance; build one and reuse it."""

    def __init__(self, api_key: str | None = None, *, rate_per_minute: float = 250.0) -> None:
        self._api_key = api_key or settings.FMP_API_KEY
        if not self._api_key:
            raise RuntimeError("FMP requires an API key. Set FMP_API_KEY in .env.")
        self._limiter = RateLimiter(rate_per_minute=rate_per_minute)
        # httpx logs every request URL at INFO, query string included.
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _get(self, url: str, **params) -> list | dict:
        """GET and decode. Every failure becomes FmpError."""
        path = url.split("?", 1)[0]
        query = {k: v for k, v in params.items() if v is not None}
        query["apikey"] = self._api_key

        try:
            raw = get(url, params=query, limiter=self._limiter, retry_statuses=RETRY_STATUSES)
        except httpx.HTTPStatusError as exc:
            raise FmpError(path, status=exc.response.status_code) from None
        except httpx.TransportError as exc:
            raise FmpError(path, reason=type(exc).__name__) from None

        try:
            body = json.loads(raw)
        except ValueError:
            raise FmpError(path, reason="response is not JSON") from None

        # FMP reports some errors as 200 with a message body.
        if isinstance(body, dict) and "Error Message" in body:
            raise FmpError(path, reason=str(body["Error Message"])[:200])
        return body

    def stock_list(self) -> list[dict]:
        """Every listing FMP knows, ~93k rows. One call."""
        return self._get(f"{V3_BASE}/stock/list")

    def screener(
        self,
        *,
        min_price: float = DEFAULT_MIN_PRICE,
        min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
        min_volume: int = DEFAULT_MIN_VOLUME,
        exchanges: tuple[str, ...] = US_EXCHANGES,
    ) -> list[dict]:
        """Actively traded stocks above the floors, filtered server-side. One call."""
        path = f"{V3_BASE}/stock-screener"
        rows = self._get(
            path,
            exchange=",".join(exchanges),
            priceMoreThan=min_price,
            marketCapMoreThan=int(min_market_cap),
            volumeMoreThan=int(min_volume),
            isActivelyTrading="true",
            isEtf="false",
            isFund="false",
            limit=SCREENER_LIMIT,
        )
        if len(rows) >= SCREENER_LIMIT:
            raise FmpError(path, reason=f"screener hit its {SCREENER_LIMIT}-row cap; results truncated")
        return rows

    def earnings_v3(self, symbol: str) -> list[dict]:
        """Full announcement history, newest first. Ignores from/to."""
        return self._get(f"{V3_BASE}/historical/earning_calendar/{_symbol_path(symbol)}")

    def earnings_stable(self, symbol: str) -> list[dict]:
        """Same history with authoritative dates, but no timing or fiscal period."""
        return self._get(f"{STABLE_BASE}/earnings", symbol=symbol, limit=STABLE_HISTORY_LIMIT)


def parse_universe(records: list[dict], *, stats: dict | None = None) -> list[str]:
    """Symbols from screener rows, largest market cap first, unique.

    Drops non-common and shell names. `stats` gets the count per stage.
    """
    common = [r for r in records if not NON_COMMON_PATTERN.search(r.get("companyName") or "")]
    operating = [r for r in common if not SHELL_PATTERN.search(r.get("companyName") or "")]

    seen: set[str] = set()
    symbols: list[str] = []
    for r in sorted(operating, key=lambda r: r.get("marketCap") or 0, reverse=True):
        symbol = r.get("symbol")
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    if stats is not None:
        stats.update(screened=len(records), common=len(common), operating=len(operating), symbols=len(symbols))
    return symbols


def parse_earnings(symbol: str, rows: list[dict]) -> list[EarningsRecord]:
    """v3 rows to records, in input order. Rows without a usable date are skipped."""
    records = []
    for row in rows:
        date = _to_date(row.get("date"))
        if date is None:
            logger.warning(f"{symbol}: skipping v3 row with bad date {row.get('date')!r}")
            continue
        time = row.get("time") or None
        records.append(
            EarningsRecord(
                ticker=symbol,  # from the request, not the row
                date=date,
                time=time,
                timing_label=TIMING_LABELS.get(time, "Unspecified"),
                date_corrected=False,
                fiscal_date_ending=_to_date(row.get("fiscalDateEnding")),
                eps_actual=_to_float(row.get("eps")),
                eps_estimated=_to_float(row.get("epsEstimated")),
                revenue_actual=_to_float(row.get("revenue")),
                revenue_estimated=_to_float(row.get("revenueEstimated")),
            )
        )
    return records


def parse_stable_dates(rows: list[dict]) -> list[datetime.date]:
    """Announcement dates from the stable endpoint, ascending."""
    return sorted(d for d in (_to_date(r.get("date")) for r in rows if isinstance(r, dict)) if d)


def reconcile_dates(
    reported: list[datetime.date],
    authoritative: list[datetime.date],
    tolerance: int = DATE_MATCH_TOLERANCE_DAYS,
) -> list[datetime.date | None]:
    """Nearest authoritative date for each reported one, aligned to `reported`.

    One-to-one and claimed in ascending order, so a quarter missing from one
    side cannot shift its neighbours. None where nothing is within tolerance.
    """
    pool = sorted(authoritative)
    claimed = [False] * len(pool)
    matched: list[datetime.date | None] = [None] * len(reported)

    for idx in sorted(range(len(reported)), key=lambda i: reported[i]):
        best_i, best_gap = None, tolerance + 1
        for i, candidate in enumerate(pool):
            if claimed[i]:
                continue
            gap = abs((candidate - reported[idx]).days)
            if gap < best_gap:
                best_i, best_gap = i, gap
        if best_i is not None:
            claimed[best_i] = True
            matched[idx] = pool[best_i]
    return matched


def apply_authoritative_dates(
    records: list[EarningsRecord], authoritative: list[datetime.date]
) -> list[EarningsRecord]:
    """Swap in stable dates; flag the rows that moved. Unmatched rows keep v3's date."""
    matched = reconcile_dates([r.date for r in records], authoritative)
    return [
        dataclasses.replace(r, date=found, date_corrected=True) if found and found != r.date else r
        for r, found in zip(records, matched)
    ]


def pct_change(actual: float | None, baseline: float | None) -> float | None:
    """Percent difference over |baseline|, so beating a negative estimate is positive."""
    if actual is None or not baseline:
        return None
    return round((actual - baseline) / abs(baseline) * 100, 2)


def _difference(actual: float | None, baseline: float | None) -> float | None:
    return None if actual is None or baseline is None else actual - baseline


def sales_growth_yoy(records: list[EarningsRecord]) -> list[float | None]:
    """YoY revenue growth per record, aligned to input.

    Prior year = the period nearest fiscal_date_ending - 365 days, within
    YOY_PERIOD_TOLERANCE_DAYS. Needs the symbol's full history.
    """
    by_period: dict[datetime.date, float] = {}
    for r in records:
        if r.fiscal_date_ending and r.revenue_actual is not None:
            by_period.setdefault(r.fiscal_date_ending, r.revenue_actual)  # first wins

    growth: list[float | None] = []
    for r in records:
        if not r.fiscal_date_ending or r.revenue_actual is None:
            growth.append(None)
            continue
        target = r.fiscal_date_ending - datetime.timedelta(days=365)
        within = [
            (abs((period - target).days), period)
            for period in by_period
            if abs((period - target).days) <= YOY_PERIOD_TOLERANCE_DAYS
        ]
        prior = by_period[min(within)[1]] if within else None
        growth.append(round((r.revenue_actual - prior) / abs(prior) * 100, 2) if prior else None)
    return growth


def with_derived_metrics(records: list[EarningsRecord]) -> list[EarningsRecord]:
    """Fill the five derived fields (ADR 0007)."""
    growth = sales_growth_yoy(records)
    return [
        dataclasses.replace(
            r,
            eps_surprise=_difference(r.eps_actual, r.eps_estimated),
            eps_surprise_pct=pct_change(r.eps_actual, r.eps_estimated),
            revenue_surprise=_difference(r.revenue_actual, r.revenue_estimated),
            revenue_surprise_pct=pct_change(r.revenue_actual, r.revenue_estimated),
            sales_growth_yoy_pct=g,
        )
        for r, g in zip(records, growth)
    ]


def dedupe(records: list[EarningsRecord]) -> list[EarningsRecord]:
    """Newest first, one record per (ticker, date); first seen wins."""
    seen: set[tuple[str, datetime.date]] = set()
    out = []
    for r in sorted(records, key=lambda r: r.date, reverse=True):
        if (r.ticker, r.date) not in seen:
            seen.add((r.ticker, r.date))
            out.append(r)
    return out


def _to_date(value: object) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    """Numbers and numeric strings to float; anything else to None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _symbol_path(symbol: str) -> str:
    return quote(symbol, safe="")
