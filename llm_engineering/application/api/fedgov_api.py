"""federalreserve.gov as a registered source.

A thin adapter over the pure functions in `fedgov`. It holds no logic of its
own -- the parsing, classification and extraction all stay in the module they
are tested in. Its only job is to give the dispatcher something to register.
"""

import itertools
from collections.abc import Iterator
from pathlib import Path

from llm_engineering.application.api import fedgov
from llm_engineering.application.api.base import BaseApi
from llm_engineering.application.api.models import FedSourceDocument
from llm_engineering.domain.documents import FedDocument
from llm_engineering.infrastructure.http import get


class FedGovApi(BaseApi):
    model = FedDocument

    #: Used by ApiDispatcher.register_fedgov() for host-based lookup.
    domain = fedgov.BASE

    def iter_documents(
        self, *, cache_dir: Path | None = None, force: bool = False
    ) -> Iterator[FedSourceDocument]:
        """Index-page documents, then press-feed statements.

        The press feed is always fetched with force=True: it is the one file
        in this pipeline that changes, and a cached copy turns an incremental
        run into a no-op.
        """
        return itertools.chain(
            fedgov.iter_documents(cache_dir=cache_dir, force=force),
            fedgov.fetch_press_index(cache_dir=cache_dir, force=True),
        )

    def fetch(self, url: str, *, cache_dir: Path | None = None) -> bytes:
        return get(url, cache_dir=cache_dir)

    def extract_text(self, content: bytes, media_type: str) -> fedgov.ExtractedText:
        return fedgov.extract_text(content, media_type)
