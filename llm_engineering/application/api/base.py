from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from llm_engineering.application.api.models import FedSourceDocument
from llm_engineering.domain.base import NoSQLBaseDocument


class BaseApi(ABC):
    """One source of FOMC documents.

    Two responsibilities, deliberately separated: enumerating what exists
    (cheap, metadata only) and fetching one document's bytes (expensive).
    The ETL runs them as separate pipeline steps so a 15-40 GB download is
    never the price of refreshing a manifest.
    """

    model: type[NoSQLBaseDocument]

    @abstractmethod
    def iter_documents(
        self,
        *,
        cache_dir: Path | None = None,
        force: bool = False,
        incremental: bool = False,
    ) -> Iterator[FedSourceDocument]:
        """Yield every document this source knows about. Metadata only.

        `incremental` asks the source to skip whatever cannot have changed.
        It is a hint, not a contract: a source with no cheap update path is
        free to ignore it and do a full pass, because the manifest step
        dedupes on stored URLs either way.
        """

    @abstractmethod
    def fetch(self, url: str, *, cache_dir: Path | None = None) -> bytes:
        """Fetch one document's raw bytes."""
