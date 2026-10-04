from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.api.dispatcher import ApiDispatcher
from llm_engineering.domain.documents import FedDocument, FedTextDocument


@step
def extract_fed_text(
    document_urls: list[str],
    year_from: int | None = None,
    year_to: int | None = None,
    cache_dir: str | None = None,
    limit: int | None = None,
) -> Annotated[int, "extracted_count"]:
    """Download and extract text for documents not already extracted.

    This is the expensive half -- the full corpus is 15-40 GB. `year_from` /
    `year_to` bound it so the first run can be one year, and `limit` caps a
    smoke test. Resumption is free: anything already in fed_text is skipped,
    so a run killed at document 3,000 picks up at 3,000.
    """
    dispatcher = ApiDispatcher.build().register_fedgov()
    # Same boundary conversion as build_fed_manifest: ZenML params are
    # JSON-serialisable, so cache_dir is a string here.
    cache = Path(cache_dir) if cache_dir else None

    wanted = set(document_urls)
    pending = [
        doc
        for doc in FedDocument.bulk_find()
        if doc.url in wanted and _in_range(doc.meeting_date.year, year_from, year_to)
    ]
    already = {text.url for text in FedTextDocument.bulk_find()}
    pending = [doc for doc in pending if doc.url not in already]
    pending.sort(key=lambda doc: doc.meeting_date)

    if limit is not None:
        pending = pending[:limit]

    logger.info(f"{len(pending)} documents to extract ({len(already)} already done)")

    extracted = 0
    empty: list[str] = []
    failed: list[str] = []

    for doc in tqdm(pending, desc="extracting"):
        api = dispatcher.get_api(doc.url)
        if api is None:
            failed.append(doc.url)
            continue

        try:
            content = api.fetch(doc.url, cache_dir=cache)
            result = api.extract_text(content, doc.media_type)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.warning(f"Fetch failed for {doc.url}: {exc!r}")
            failed.append(doc.url)
            continue
        except Exception as exc:  # a malformed PDF must not end the run
            logger.warning(f"Extraction failed for {doc.url}: {exc!r}")
            failed.append(doc.url)
            continue

        if result.is_empty:
            # Almost always a scan with no OCR layer. Recorded, not stored:
            # an empty fed_text row would make it look done on the next run.
            empty.append(doc.url)
            continue

        saved = FedTextDocument(
            url=doc.url,
            text=result.text,
            char_count=len(result.text),
            page_chars=list(result.page_chars),
            failed_pages=list(result.failed_pages),
        ).save()

        if saved is None:
            failed.append(doc.url)
            continue

        extracted += 1

    logger.info(f"Extracted {extracted}, empty {len(empty)}, failed {len(failed)}")

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="extracted_count",
        metadata=_get_metadata(pending, extracted, empty, failed, year_from, year_to),
    )

    return extracted


def _in_range(year: int, year_from: int | None, year_to: int | None) -> bool:
    if year_from is not None and year < year_from:
        return False
    if year_to is not None and year > year_to:
        return False
    return True


def _get_metadata(
    pending: list[FedDocument],
    extracted: int,
    empty: list[str],
    failed: list[str],
    year_from: int | None,
    year_to: int | None,
) -> dict:
    return {
        "query": {"year_from": year_from, "year_to": year_to, "candidates": len(pending)},
        "retrieved": {
            "extracted": extracted,
            "empty": len(empty),
            "failed": len(failed),
            "empty_sample": empty[:5],
            "failed_sample": failed[:5],
        },
    }
