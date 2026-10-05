from collections import Counter
from pathlib import Path

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.api.dispatcher import ApiDispatcher
from llm_engineering.application.api.models import FedSourceDocument
from llm_engineering.domain.documents import FedDocument, to_bson_datetime


@step
def build_fed_manifest(
    sources: list[str],
    cache_dir: str | None = None,
    force: bool = False,
) -> Annotated[list[str], "document_urls"]:
    """Enumerate every known FOMC document and store its metadata.

    Metadata only -- roughly 90 fetches for fed.gov's ~5,200 documents, 11
    API calls for FRASER's 581. Nothing is downloaded here, so this is safe
    to re-run and cheap enough to schedule.

    `sources` comes from the config, which is what lets one pipeline serve
    both corpora: fed_data_etl.yaml says [fed_gov], fraser_data_etl.yaml says
    [fraser]. Rows carry `source`, so the two never need merging -- precedence
    is a retrieval-time decision, taken downstream where it stays reversible.

    Idempotent by pre-filtering on URLs already in Mongo. NoSQLBaseDocument
    has no upsert, so the alternative would be duplicate inserts on every run.
    """
    dispatcher = ApiDispatcher.from_sources(sources)
    # ZenML parameters must be JSON-serialisable, so cache_dir arrives as a
    # string. Widening to Path belongs here, at the boundary.
    cache = Path(cache_dir) if cache_dir else None

    discovered: dict[str, FedSourceDocument] = {}
    for api in dispatcher.sources():
        for document in api.iter_documents(cache_dir=cache, force=force):
            discovered.setdefault(document.url, document)

    # Deliberately every stored URL, not just this run's sources. URLs are
    # unique across sources (different hosts), so a global set costs nothing
    # and makes a mis-scoped `sources` list unable to double-insert.
    known = {document.url for document in FedDocument.bulk_find()}
    new = [doc for url, doc in discovered.items() if url not in known]

    logger.info(
        f"{sources}: discovered {len(discovered)}, "
        f"already stored {len(known)} (all sources), new {len(new)}"
    )

    if new:
        # bulk_insert swallows BulkWriteError and returns False, so an
        # unchecked call makes a failed batch look like a successful one.
        if not FedDocument.bulk_insert([_to_document(doc) for doc in new]):
            raise RuntimeError(f"Failed to insert {len(new)} documents into fed_documents")

    urls = sorted(discovered)
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="document_urls", metadata=_get_metadata(sources, discovered, known, new)
    )

    return urls


def _to_document(source: FedSourceDocument) -> FedDocument:
    return FedDocument(
        source=source.source,
        info_type=source.info_type,
        meeting_date=to_bson_datetime(source.meeting_date),
        title=source.title,
        url=source.url,
        media_type=source.media_type,
        text_url=source.text_url,
        source_id=source.source_id,
        page_count=source.page_count,
    )


def _get_metadata(
    sources: list[str],
    discovered: dict[str, FedSourceDocument],
    known: set[str],
    new: list[FedSourceDocument],
) -> dict:
    dates = [doc.meeting_date for doc in discovered.values()]
    return {
        "query": {"sources": sources},
        "retrieved": {
            "discovered": len(discovered),
            "already_stored": len(known),
            "inserted": len(new),
            "earliest": str(min(dates)) if dates else None,
            "latest": str(max(dates)) if dates else None,
            "by_info_type": dict(
                Counter(doc.info_type.value for doc in discovered.values()).most_common()
            ),
        },
    }
