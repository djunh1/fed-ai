import re
from urllib.parse import urlparse

from loguru import logger

from .base import BaseApi
from .fedgov_api import FedGovApi

_LEADING_WWW = re.compile(r"^www\.", re.IGNORECASE)


class ApiDispatcher:
    """Registry of document sources, keyed by host.

    Built fluently, the way the handbook's CrawlerDispatcher is:

        ApiDispatcher.build().register_fedgov()

    Two different lookups hang off this. The manifest step walks `sources()`
    to enumerate everything; the text step calls `get_api(url)` to route one
    document back to the source that knows how to fetch it.
    """

    def __init__(self) -> None:
        self._apis: dict[str, BaseApi] = {}

    @classmethod
    def build(cls) -> "ApiDispatcher":
        return cls()

    def register(self, domain: str, api: BaseApi) -> "ApiDispatcher":
        self._apis[_host(domain)] = api
        return self

    def register_fedgov(self) -> "ApiDispatcher":
        return self.register(FedGovApi.domain, FedGovApi())

    def get_api(self, url: str) -> BaseApi | None:
        """The source that owns this URL, or None.

        None rather than an exception: a manifest built by an older run can
        contain hosts this dispatcher no longer registers, and one unroutable
        URL should not end a harvest.
        """
        api = self._apis.get(_host(url))
        if api is None:
            logger.warning(f"No API registered for {url}")
        return api

    def sources(self) -> list[BaseApi]:
        return list(self._apis.values())

    def __len__(self) -> int:
        return len(self._apis)


def _host(url: str) -> str:
    """Normalise a URL or bare domain down to a comparable host.

    Accepts both because register() is called with domains
    ("https://www.federalreserve.gov") and get_api() with full document URLs.
    The www. strip is applied to the netloc, not the whole URL -- anchoring it
    to the scheme means it never fires once urlparse has removed the scheme.
    """
    netloc = urlparse(url).netloc or url
    netloc = netloc.rsplit("@", 1)[-1].split(":", 1)[0]
    return _LEADING_WWW.sub("", netloc).strip("/").lower()
