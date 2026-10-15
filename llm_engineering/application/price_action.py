"""Market reaction to one announcement, ported from the sister project's
PriceActionService (ADR 0008). N_d is trading days: FMP returns only sessions
that traded, so stepping through the bars is the calendar."""

import bisect
import datetime
import math
from dataclasses import dataclass

#: Close of day N vs day-0 open: `{N}_d`.
RETURN_HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60)
#: Highest high in days 0..N vs day-0 open. 70 is from sample_data.csv.
MAX_GAIN_HORIZONS = (10, 20, 40, 60, 70)
#: Lowest low in days 0..N vs day-0 open, signed negative.
MAX_LOSS_HORIZONS = (10, 20, 40, 60)

#: Sizes fetch windows only; ~252 sessions per 365 days.
CALENDAR_PER_TRADING_DAY = 1.45
#: The market-cap endpoint truncates at ~1,300 rows; fetch in windows under it.
MARKET_CAP_CHUNK_DAYS = 730
#: Calendar days fetched before the first announcement: the prior close and
#: the day-0 check's ATR (ATR_SESSIONS + 2 sessions).
LOOKBACK_DAYS = 30
#: Longest legitimate wait for day 0: Thu amc before a long weekend is 4.
#: Beyond this the stock was not trading (pre-IPO, halted); no anchor. The
#: sister has no cap, so its pre-IPO rows carry the IPO's move (ADR 0008).
MAX_ANCHOR_GAP_DAYS = 5

#: Day-0 check: ATR over this many sessions, ending 2 before day 0.
ATR_SESSIONS = 3
#: A neighbour's gap above this many ATRs, and above day 0's, is suspect.
SUSPECT_GAP_ATRS = 2.0

PRICE_COLUMNS = (
    "open_0d", "close_0d", "high_0d", "low_0d", "volume",
    "percent_gap_up", "c-pc_pct", "max_gain_0d", "max_down_intraday",
)  # fmt: skip
PROFILE_COLUMNS = ("sector", "industry", "market_capitalization")

#: Everything price_action_for() returns, in output order.
PRICE_ACTION_COLUMNS: tuple[str, ...] = (
    ("date_0d", "anchor_found", "day_0_check")
    + PRICE_COLUMNS
    + tuple(f"{n}_d" for n in RETURN_HORIZONS)
    + tuple(f"{n}_d_max_gain" for n in MAX_GAIN_HORIZONS)
    + tuple(f"{n}_d_max_draw" for n in MAX_LOSS_HORIZONS)
    + PROFILE_COLUMNS
)

#: Filled last; a row with this set has every horizon it will ever have.
LONGEST_HORIZON_COLUMN = f"{max(RETURN_HORIZONS + MAX_GAIN_HORIZONS + MAX_LOSS_HORIZONS)}_d_max_gain"


@dataclass(frozen=True, slots=True)
class Bar:
    date: datetime.date
    open: float
    high: float
    low: float
    close: float
    volume: float


def trading_to_calendar_days(trading_days: int, safety: int = 5) -> int:
    return math.ceil(trading_days * CALENDAR_PER_TRADING_DAY) + safety


def lookforward_days() -> int:
    """Calendar days that cover the longest horizon (~107)."""
    return trading_to_calendar_days(max(RETURN_HORIZONS + MAX_GAIN_HORIZONS + MAX_LOSS_HORIZONS))


def parse_bars(payload: object) -> list[Bar]:
    """historical-price-full payload to bars, ascending, one per date. Bad rows skipped."""
    rows = payload.get("historical") if isinstance(payload, dict) else None
    bars: dict[datetime.date, Bar] = {}
    for row in rows or []:
        try:
            bar = Bar(
                date=datetime.date.fromisoformat(str(row["date"])[:10]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        bars.setdefault(bar.date, bar)
    return [bars[d] for d in sorted(bars)]


def parse_market_caps(rows: list[dict]) -> list[tuple[datetime.date, float]]:
    """Daily market caps, ascending, one per date."""
    caps: dict[datetime.date, float] = {}
    for row in rows:
        try:
            caps.setdefault(
                datetime.date.fromisoformat(str(row["date"])[:10]),
                float(row["marketCap"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(caps.items())


def parse_profile(payload: object) -> dict:
    record = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else {}
    return {
        "sector": record.get("sector") or None,
        "industry": record.get("industry") or None,
    }


def market_cap_windows(start: datetime.date, end: datetime.date) -> list[tuple[datetime.date, datetime.date]]:
    windows = []
    while start <= end:
        stop = min(start + datetime.timedelta(days=MARKET_CAP_CHUNK_DAYS), end)
        windows.append((start, stop))
        start = stop + datetime.timedelta(days=1)
    return windows


def market_cap_as_of(caps: list[tuple[datetime.date, float]], as_of: datetime.date) -> float | None:
    """Cap on `as_of`, else the latest before it. Historical, not today's (no lookahead)."""
    i = bisect.bisect_right(caps, (as_of, math.inf))
    return caps[i - 1][1] if i else None


def find_anchor_index(bars: list[Bar], announce_date: datetime.date, timing: str | None) -> int | None:
    """First session the market could react in. bmo: same day or later; anything
    else (amc, dmh, unknown) treated as after the close: next session.
    None if no session within MAX_ANCHOR_GAP_DAYS."""
    dates = [b.date for b in bars]
    if str(timing).lower() == "bmo":
        i = bisect.bisect_left(dates, announce_date)
    else:
        i = bisect.bisect_right(dates, announce_date)
    if i == len(bars) or (dates[i] - announce_date).days > MAX_ANCHOR_GAP_DAYS:
        return None
    return i


def compute_row_metrics(bars: list[Bar], anchor: int) -> dict:
    """All moves vs the day-0 open. Horizons past the data are left out."""
    session = bars[anchor]
    open_0d = session.open
    metrics: dict = {
        "date_0d": session.date,
        "open_0d": open_0d,
        "close_0d": session.close,
        "high_0d": session.high,
        "low_0d": session.low,
        "volume": session.volume,
    }
    if open_0d == 0:
        return metrics

    prev_close = bars[anchor - 1].close if anchor > 0 else None
    if prev_close:
        metrics["percent_gap_up"] = _pct(open_0d, prev_close)
        metrics["c-pc_pct"] = _pct(session.close, prev_close)
    metrics["max_gain_0d"] = _pct(session.high, open_0d)
    metrics["max_down_intraday"] = _pct(session.low, open_0d)

    last = len(bars) - 1
    for n in RETURN_HORIZONS:
        if anchor + n <= last:
            metrics[f"{n}_d"] = _pct(bars[anchor + n].close, open_0d)
    for n in MAX_GAIN_HORIZONS:
        if anchor + n <= last:
            metrics[f"{n}_d_max_gain"] = _pct(max(b.high for b in bars[anchor : anchor + n + 1]), open_0d)
    for n in MAX_LOSS_HORIZONS:
        if anchor + n <= last:
            metrics[f"{n}_d_max_draw"] = _pct(min(b.low for b in bars[anchor : anchor + n + 1]), open_0d)
    return metrics


def check_day_0(bars: list[Bar], anchor: int) -> str | None:
    """ "late"/"early" if the session before/after day 0 gapped more than
    SUSPECT_GAP_ATRS and more than day 0 did; else "ok". None if too little
    history or no next session. Informational: nothing is corrected."""
    if anchor < ATR_SESSIONS + 2 or anchor + 1 >= len(bars):
        return None
    ranges = [
        max(b.high, prev.close) - min(b.low, prev.close)
        for prev, b in zip(
            bars[anchor - ATR_SESSIONS - 2 : anchor - 2],
            bars[anchor - ATR_SESSIONS - 1 : anchor - 1],
        )
    ]
    atr = sum(ranges) / len(ranges)
    if atr <= 0:
        return None

    def gap(i: int) -> float:
        return abs(bars[i].open - bars[i - 1].close) / atr

    day_0 = gap(anchor)
    late, early = gap(anchor - 1), gap(anchor + 1)
    if late > SUSPECT_GAP_ATRS and late > day_0 and late >= early:
        return "late"
    if early > SUSPECT_GAP_ATRS and early > day_0:
        return "early"
    return "ok"


def price_action_for(
    bars: list[Bar],
    caps: list[tuple[datetime.date, float]],
    profile: dict,
    announce_date: datetime.date,
    timing: str | None,
) -> dict:
    """Every PRICE_ACTION_COLUMNS key for one announcement; None where unknown."""
    out = dict.fromkeys(PRICE_ACTION_COLUMNS)
    anchor = find_anchor_index(bars, announce_date, timing)
    out["anchor_found"] = anchor is not None
    if anchor is not None:
        out.update(compute_row_metrics(bars, anchor))
        out["day_0_check"] = check_day_0(bars, anchor)
        out["market_capitalization"] = market_cap_as_of(caps, out["date_0d"])
    out["sector"] = profile.get("sector")
    out["industry"] = profile.get("industry")
    return out


def _pct(value: float, base: float) -> float:
    return round((value - base) / base * 100, 2)
