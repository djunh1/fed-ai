"""Unit tests for llm_engineering.infrastructure.http.RateLimiter.

A rate limiter is defined entirely in terms of time, which makes it awkward to
test honestly: asserting "28 requests took about 60 seconds" means a one-minute
test that is still flaky on a loaded machine.

So we replace the clock. `FakeClock` provides both `monotonic` and `sleep`, and
sleeping advances the clock instead of blocking. That gives us a simulated
timeline we can assert to the microsecond, and the whole suite runs instantly.

Most tests use rate_per_minute=60.0 so the rate is exactly 1 token/second and
the arithmetic stays checkable by eye.

Run: uv run pytest tests/unit/rate_limiter_test.py -v
"""

import time

import pytest

from llm_engineering.infrastructure.http import RateLimiter

# 1 token per second, so "seconds elapsed" and "tokens earned" are the same number.
ONE_PER_SECOND = 60.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class FakeClock:
    """A controllable stand-in for time.monotonic and time.sleep.

    Sleeping advances the clock rather than blocking, which is what a real
    sleep does from the program's point of view -- just without the waiting.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration

    def advance(self, seconds: float) -> None:
        """Simulate time passing for reasons other than a sleep."""
        self.now += seconds

    @property
    def total_slept(self) -> float:
        return sum(self.sleeps)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Install a FakeClock over the time module for the duration of a test."""
    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake.monotonic)
    monkeypatch.setattr(time, "sleep", fake.sleep)
    return fake


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_starts_with_a_full_bucket(clock: FakeClock) -> None:
    """A fresh limiter has used no quota, so it must not make anyone wait."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=5.0)

    for _ in range(5):
        limiter.acquire()

    assert clock.sleeps == []


def test_rate_is_converted_from_per_minute_to_per_second(clock: FakeClock) -> None:
    """30/min is 0.5/s, so an empty bucket earns a token in 2 seconds."""
    limiter = RateLimiter(rate_per_minute=30.0, capacity=1.0)

    limiter.acquire()  # spends the initial token
    limiter.acquire()  # must wait a full token's worth

    assert clock.sleeps == pytest.approx([2.0])


def test_defaults_leave_headroom_under_frasers_thirty_per_minute(clock: FakeClock) -> None:
    """The margin is deliberate: our minute and the server's do not align, so
    running at exactly 30/min will occasionally be counted as 31."""
    limiter = RateLimiter()

    assert limiter._rate * 60.0 < 30.0


# --------------------------------------------------------------------------
# the fast path
# --------------------------------------------------------------------------


def test_acquire_returns_nothing(clock: FakeClock) -> None:
    """It enforces the delay rather than reporting it. A returned float that a
    caller forgets to honour is a limiter that does nothing."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=2.0)

    assert limiter.acquire() is None


def test_spare_tokens_are_spent_without_sleeping(clock: FakeClock) -> None:
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=3.0)

    limiter.acquire()

    assert clock.sleeps == []
    assert limiter._tokens == pytest.approx(2.0)


def test_time_passing_refills_the_bucket(clock: FakeClock) -> None:
    """Refill is computed lazily from elapsed time -- no timer, no thread."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=5.0)
    for _ in range(5):
        limiter.acquire()

    clock.advance(3.0)
    for _ in range(3):
        limiter.acquire()

    assert clock.sleeps == [], "3 seconds at 1/s earns exactly the 3 tokens spent"


# --------------------------------------------------------------------------
# the slow path
# --------------------------------------------------------------------------


def test_an_empty_bucket_blocks(clock: FakeClock) -> None:
    """The bug that hides best: computing the wait and then not sleeping. The
    limiter silently stops limiting exactly when it should start."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=1.0)

    limiter.acquire()
    limiter.acquire()

    assert clock.sleeps == pytest.approx([1.0])


def test_sleep_covers_only_the_shortfall(clock: FakeClock) -> None:
    """With 0.25 tokens banked you need 0.75 more, not a whole one."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=1.0)
    limiter.acquire()
    clock.advance(0.25)

    limiter.acquire()

    assert clock.sleeps == pytest.approx([0.75])


def test_tokens_are_zeroed_after_waiting(clock: FakeClock) -> None:
    """You slept until you had exactly one token, then spent it. Subtracting
    instead would leave the count negative and skew every later refill."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=1.0)
    limiter.acquire()

    limiter.acquire()

    assert limiter._tokens == pytest.approx(0.0)


def test_clock_is_reread_after_sleeping(clock: FakeClock) -> None:
    """Reusing the pre-sleep timestamp credits the limiter for time it spent
    sleeping, silently doubling the effective rate.

    At 1/s with capacity 1: acquire, acquire (sleep 1s), acquire (sleep 1s).
    Correct implementations land at t+2. The stale-timestamp bug lands at t+1,
    because the second call's refill double-counts the first call's sleep.
    """
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=1.0)
    started = clock.now

    for _ in range(3):
        limiter.acquire()

    assert clock.now - started == pytest.approx(2.0)


def test_sustained_rate_matches_the_configured_rate(clock: FakeClock) -> None:
    """The end-to-end property. 20 requests, 5 of them free from the initial
    bucket, leaves 15 to be earned at 1/s."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=5.0)
    started = clock.now

    for _ in range(20):
        limiter.acquire()

    assert clock.now - started == pytest.approx(15.0)


def test_eleven_fraser_calls_stay_under_a_minute(clock: FakeClock) -> None:
    """Paginating title 677 at limit=100 takes 11 calls. At the real defaults
    that should cost only a few seconds, not a stall."""
    limiter = RateLimiter()
    started = clock.now

    for _ in range(11):
        limiter.acquire()

    assert clock.now - started < 20.0


# --------------------------------------------------------------------------
# the capacity clamp
# --------------------------------------------------------------------------


def test_idling_does_not_bank_unlimited_tokens(clock: FakeClock) -> None:
    """Without the min() clamp, an hour of idling earns 3600 tokens and the
    next burst ignores the limit entirely."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=5.0)
    clock.advance(3600.0)

    for _ in range(5):
        limiter.acquire()

    assert limiter._tokens == pytest.approx(0.0)


def test_bursts_are_bounded_by_capacity(clock: FakeClock) -> None:
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=5.0)
    clock.advance(3600.0)

    for _ in range(6):
        limiter.acquire()

    assert len(clock.sleeps) == 1, "only the 6th call should have waited"


def test_capacity_of_one_removes_bursting(clock: FakeClock) -> None:
    """A strictly-paced limiter: every call after the first pays full price."""
    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=1.0)

    for _ in range(4):
        limiter.acquire()

    assert clock.sleeps == pytest.approx([1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# clock choice
# --------------------------------------------------------------------------


def test_wall_clock_is_never_consulted(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    """time.time() can be stepped backwards by NTP mid-run, which makes elapsed
    time negative. Booby-trap it to prove the limiter uses monotonic only."""

    def explode() -> float:
        raise AssertionError("RateLimiter must use time.monotonic, not time.time")

    monkeypatch.setattr(time, "time", explode)

    limiter = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=1.0)
    limiter.acquire()
    limiter.acquire()


# --------------------------------------------------------------------------
# independence
# --------------------------------------------------------------------------


def test_limiters_do_not_share_state(clock: FakeClock) -> None:
    """State lives on the instance. A mutable class attribute would make every
    FRASER client in the process share one bucket."""
    a = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=2.0)
    b = RateLimiter(rate_per_minute=ONE_PER_SECOND, capacity=2.0)

    a.acquire()
    a.acquire()

    assert b._tokens == pytest.approx(2.0)
    b.acquire()
    assert clock.sleeps == []
