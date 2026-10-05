import re
from collections.abc import Iterable
from urllib.parse import urlparse

from loguru import logger

from llm_engineering.domain.types import SourceSystem

from .base import BaseApi
from .fedgov_api import FedGovApi
from .fraser_api import FraserApi

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

    @classmethod
    def from_sources(cls, sources: Iterable[str]) -> "ApiDispatcher":
        """Build from SourceSystem values, as the pipeline config names them.

        This is what lets one pipeline serve both corpora: the config says
        `sources: [fed_gov]` or `sources: [fraser]`, and the steps need no
        knowledge of which is which.

        An unknown name raises rather than being skipped. A typo in the
        config would otherwise produce an empty dispatcher and a run that
        reports "0 new documents" as a success.
        """
        dispatcher = cls()
        for name in sources:
            register = _REGISTRARS.get(SourceSystem(name))
            if register is None:
                raise ValueError(f"No API registered for source {name!r}")
            register(dispatcher)
        return dispatcher

    def register(self, domain: str, api: BaseApi) -> "ApiDispatcher":
        self._apis[_host(domain)] = api
        return self

    def register_fedgov(self) -> "ApiDispatcher":
        return self.register(FedGovApi.domain, FedGovApi())

    def register_fraser(self) -> "ApiDispatcher":
        return self.register(FraserApi.domain, FraserApi())

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


#: SourceSystem to the fluent registrar that adds it. Defined after the class
#: because the values are unbound methods of it.
_REGISTRARS = {
    SourceSystem.FED_GOV: ApiDispatcher.register_fedgov,
    SourceSystem.FRASER: ApiDispatcher.register_fraser,
}


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
