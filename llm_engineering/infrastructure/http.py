from pathlib import Path
import httpx
import hashlib
import json
import time

from loguru import logger

MAX_ATTEMPTS = 3

class RateLimiter:
    """
    Rate limiter for the FRASER Api
    """

    def __init__(self, rate_per_minute: float = 28.0, capacity: float = 5.0) -> None:
        self._rate = rate_per_minute / 60.0
        self._capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()

    def acquire(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now-self._last) * self._rate)

        self._last = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 
        
        wait = (1.0 - self._tokens) / self._rate
        time.sleep(wait)
        self._tokens = 0.0
        self._last = time.monotonic()
    

def _cache_key(url: str, params: dict | None) -> str:
    key = url + json.dumps(params, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()

def get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
    cache_dir: Path | None = None,  # None = no caching
    force: bool = False,
) -> bytes:

    # Read cache
    if cache_dir is not None:
        path = cache_dir / f"{_cache_key(url, params)}.bin"
        if not force and path.exists():
            return path.read_bytes()
            

    for attempt in range(MAX_ATTEMPTS):
        last_attempt = attempt == MAX_ATTEMPTS -1
        
        try:
            response = httpx.get(url, params=params, headers=headers, timeout=timeout)
        except httpx.TransportError:
            logger.warning(f"HTTP Transport error for url= {url}, on attempt {attempt}, retrying connection.")
            if last_attempt: raise
            time.sleep(2 ** attempt)
            continue

        retryable = response.status_code == 429 or response.status_code >= 500

        if not retryable or last_attempt:
            break

        retry_after = response.headers.get("Retry-After")
        try:
            delay = int(retry_after)
        except (TypeError, ValueError):
            delay = 2 ** attempt

        logger.warning(f"Retrying {url} after {delay}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
        time.sleep(delay)

    response.raise_for_status()

    # Write 
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"url": url, "params": params}, indent=2)
        )
    
    return response.content
