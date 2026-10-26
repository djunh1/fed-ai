"""Unit tests for llm_engineering.infrastructure.http.

These mirror the checks in temp/debug_http.py, but with two differences that make
them unit tests rather than a debug session:

  * No network. `httpx.get` is replaced with a stub, so the tests are fast,
    deterministic, and pass on a plane.
  * No real sleeping. `time.sleep` is replaced with a recorder, so we can assert
    the backoff schedule exactly instead of timing it.

Run: uv run pytest tests/unit/http_test.py -v
"""

import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest

from llm_engineering.infrastructure import http
from llm_engineering.infrastructure.http import MAX_ATTEMPTS, _cache_key, get

URL = "https://example.org/thing.json"
BODY = b'{"ok": true}'


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _response(status: int, *, content: bytes = BODY, headers: dict | None = None) -> httpx.Response:
    """Build a Response with a request attached.

    raise_for_status() needs .request set, otherwise it raises RuntimeError
    instead of HTTPStatusError and the test failure is confusing.
    """
    return httpx.Response(
        status_code=status,
        content=content,
        headers=headers or {},
        request=httpx.Request("GET", URL),
    )


class FakeGet:
    """Stands in for httpx.get. Returns queued results and records every call."""

    def __init__(self, *results: httpx.Response | Exception) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        # The last queued result repeats, so a single 503 means "always 503".
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def count(self) -> int:
        return len(self.calls)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Swallow real sleeps, record the durations requested."""
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def fake_get(monkeypatch: pytest.MonkeyPatch):
    """Install a FakeGet in place of httpx.get, returning the installer."""

    def install(*results: httpx.Response | Exception) -> FakeGet:
        fake = FakeGet(*results)
        monkeypatch.setattr(http.httpx, "get", fake)
        return fake

    return install


# --------------------------------------------------------------------------
# _cache_key
# --------------------------------------------------------------------------


def test_cache_key_is_a_hex_digest() -> None:
    key = _cache_key(URL, None)

    assert isinstance(key, str)
    assert len(key) == 64
    int(key, 16)  # raises ValueError if it is not hex


def test_cache_key_is_stable_across_calls() -> None:
    assert _cache_key(URL, None) == _cache_key(URL, None)


def test_cache_key_varies_with_url() -> None:
    assert _cache_key(URL, None) != _cache_key("https://example.org/other", None)


def test_cache_key_varies_with_params() -> None:
    assert _cache_key(URL, {"page": 1}) != _cache_key(URL, {"page": 2})


def test_cache_key_ignores_dict_ordering() -> None:
    """The whole reason for sort_keys=True. Without it the cache never hits."""
    assert _cache_key(URL, {"limit": 5, "page": 1}) == _cache_key(URL, {"page": 1, "limit": 5})


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_returns_response_bytes(fake_get) -> None:
    fake_get(_response(200))

    assert get(URL) == BODY


def test_success_fetches_exactly_once(fake_get) -> None:
    """A 200 must leave the loop. Falling through refetches MAX_ATTEMPTS times."""
    fake = fake_get(_response(200))

    get(URL)

    assert fake.count == 1


def test_passes_params_headers_and_timeout_through(fake_get) -> None:
    fake = fake_get(_response(200))

    get(URL, params={"page": 1}, headers={"X-Api-Key": "k"}, timeout=5.0)

    call = fake.calls[0]
    assert call["params"] == {"page": 1}
    assert call["headers"] == {"X-Api-Key": "k"}
    assert call["timeout"] == 5.0


# --------------------------------------------------------------------------
# non-retryable statuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_client_errors_raise_without_retrying(fake_get, sleeps, status: int) -> None:
    """403 is the interesting case: it is in no explicit list, so an
    implementation that enumerates non-retryable codes will loop on it."""
    fake = fake_get(_response(status))

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        get(URL)

    assert excinfo.value.response.status_code == status
    assert fake.count == 1
    assert sleeps == []


# --------------------------------------------------------------------------
# retryable statuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_statuses_use_all_attempts_then_raise(fake_get, sleeps, status: int) -> None:
    fake = fake_get(_response(status))

    with pytest.raises(httpx.HTTPStatusError):
        get(URL)

    assert fake.count == MAX_ATTEMPTS


def test_backoff_is_exponential_and_skips_the_last_attempt(fake_get, sleeps) -> None:
    """Three attempts means two waits: 2**0 then 2**1. Sleeping after the final
    attempt would just delay the exception."""
    fake_get(_response(503))

    with pytest.raises(httpx.HTTPStatusError):
        get(URL)

    assert sleeps == [1, 2]


def test_recovers_when_a_retry_succeeds(fake_get, sleeps) -> None:
    fake = fake_get(_response(503), _response(200))

    assert get(URL) == BODY
    assert fake.count == 2
    assert sleeps == [1]


def test_retry_after_header_is_honoured(fake_get, sleeps) -> None:
    """If the server says how long to wait, guessing is worse than obeying."""
    fake_get(_response(503, headers={"Retry-After": "7"}), _response(200))

    get(URL)

    assert sleeps == [7]


def test_missing_retry_after_falls_back_to_backoff(fake_get, sleeps) -> None:
    """int(None) raises TypeError, not ValueError. Most 503s send no header."""
    fake_get(_response(503, headers={}), _response(200))

    get(URL)

    assert sleeps == [1]


def test_unparseable_retry_after_falls_back_to_backoff(fake_get, sleeps) -> None:
    """The spec also permits an HTTP date, which int() cannot parse."""
    fake_get(
        _response(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        _response(200),
    )

    get(URL)

    assert sleeps == [1]


# --------------------------------------------------------------------------
# retry_statuses
# --------------------------------------------------------------------------


def test_listed_status_is_retried(fake_get, sleeps) -> None:
    """FMP throttles with 401; callers can opt it in."""
    fake = fake_get(_response(401))

    with pytest.raises(httpx.HTTPStatusError):
        get(URL, retry_statuses=(401, 429))

    assert fake.count == MAX_ATTEMPTS
    assert sleeps == [1, 2]


def test_listed_status_recovers_when_a_retry_succeeds(fake_get, sleeps) -> None:
    fake = fake_get(_response(401), _response(200))

    assert get(URL, retry_statuses=(401,)) == BODY
    assert fake.count == 2


def test_unlisted_429_is_not_retried_when_the_list_omits_it(fake_get, sleeps) -> None:
    """Replaces the default, does not extend it."""
    fake = fake_get(_response(429))

    with pytest.raises(httpx.HTTPStatusError):
        get(URL, retry_statuses=())

    assert fake.count == 1
    assert sleeps == []


def test_server_errors_are_retried_regardless_of_the_list(fake_get, sleeps) -> None:
    fake = fake_get(_response(503))

    with pytest.raises(httpx.HTTPStatusError):
        get(URL, retry_statuses=())

    assert fake.count == MAX_ATTEMPTS


# --------------------------------------------------------------------------
# transport failures
# --------------------------------------------------------------------------


def test_transport_error_retries_then_reraises(fake_get, sleeps) -> None:
    """There is no response object here, so this path must raise rather than
    fall through to raise_for_status()."""
    fake = fake_get(httpx.ConnectError("dns went away"))

    with pytest.raises(httpx.TransportError):
        get(URL)

    assert fake.count == MAX_ATTEMPTS
    assert sleeps == [1, 2]


def test_transport_error_recovers_if_a_retry_connects(fake_get, sleeps) -> None:
    fake = fake_get(httpx.ConnectError("blip"), _response(200))

    assert get(URL) == BODY
    assert fake.count == 2


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------


def test_cache_miss_fetches_and_writes(fake_get, tmp_path: Path) -> None:
    fake = fake_get(_response(200))

    body = get(URL, cache_dir=tmp_path)

    expected = tmp_path / f"{_cache_key(URL, None)}.bin"
    assert body == BODY
    assert expected.read_bytes() == BODY
    assert fake.count == 1


def test_cache_hit_does_not_touch_the_network(fake_get, tmp_path: Path) -> None:
    fake = fake_get(_response(200))

    first = get(URL, cache_dir=tmp_path)
    second = get(URL, cache_dir=tmp_path)

    assert first == second == BODY
    assert fake.count == 1, "second call should have been served from disk"


def test_cache_writes_a_readable_sidecar(fake_get, tmp_path: Path) -> None:
    """Hashed filenames are opaque; the sidecar is how you identify an entry."""
    fake_get(_response(200))

    get(URL, params={"page": 2}, cache_dir=tmp_path)

    sidecar = tmp_path / f"{_cache_key(URL, {'page': 2})}.meta.json"
    assert json.loads(sidecar.read_text()) == {"url": URL, "params": {"page": 2}}


def test_force_refetches_and_overwrites(fake_get, tmp_path: Path) -> None:
    fake = fake_get(_response(200, content=b"old"), _response(200, content=b"new"))

    first = get(URL, cache_dir=tmp_path)
    second = get(URL, cache_dir=tmp_path, force=True)

    assert (first, second) == (b"old", b"new")
    assert fake.count == 2
    assert (tmp_path / f"{_cache_key(URL, None)}.bin").read_bytes() == b"new"


def test_different_params_get_separate_cache_entries(fake_get, tmp_path: Path) -> None:
    fake = fake_get(_response(200, content=b"p1"), _response(200, content=b"p2"))

    a = get(URL, params={"page": 1}, cache_dir=tmp_path)
    b = get(URL, params={"page": 2}, cache_dir=tmp_path)

    assert (a, b) == (b"p1", b"p2")
    assert fake.count == 2
    assert len(list(tmp_path.glob("*.bin"))) == 2


def test_failures_are_not_cached(fake_get, tmp_path: Path) -> None:
    """Caching a 404 would make a transient failure permanent until someone
    manually cleared the directory."""
    fake_get(_response(404))

    with pytest.raises(httpx.HTTPStatusError):
        get(URL, cache_dir=tmp_path)

    assert list(tmp_path.glob("*.bin")) == []


def test_cache_dir_is_created_if_absent(fake_get, tmp_path: Path) -> None:
    fake_get(_response(200))
    nested = tmp_path / "a" / "b" / "c"

    get(URL, cache_dir=nested)

    assert (nested / f"{_cache_key(URL, None)}.bin").exists()


def test_no_cache_dir_means_no_files_anywhere(fake_get, tmp_path: Path) -> None:
    fake = fake_get(_response(200), _response(200))

    get(URL)
    get(URL)

    assert fake.count == 2, "without a cache_dir every call must hit the network"
    assert list(tmp_path.iterdir()) == []


def test_cache_key_matches_an_independent_calculation() -> None:
    """Pins the key derivation. If this breaks, every existing cache entry is
    orphaned -- which is fine, but should be a deliberate choice."""
    expected = hashlib.sha256((URL + json.dumps(None, sort_keys=True)).encode()).hexdigest()

    assert _cache_key(URL, None) == expected
