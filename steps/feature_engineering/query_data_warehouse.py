from collections import Counter

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.domain.documents import FedDocument, FedTextDocument


@step
def query_data_warehouse(
    fed_doc_types: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = FedDocument.bulk_find(info_type={"$in": fed_doc_types})
    fed_text = FedTextDocument.bulk_find(url={"$in": [doc.url for doc in documents]})

    by_url = {doc.url: doc for doc in documents}
    raw_documents = [
        text.model_copy(
            update={
                "info_type": doc.info_type,
                "source": doc.source,
                "meeting_date": doc.meeting_date,
                "title": doc.title,
            }
        )
        for text in fed_text
        if (doc := by_url[text.url])
    ]

    logger.info(f"{len(documents)} documents, {len(fed_text)} with text")

    get_step_context().add_output_metadata(
        output_name="raw_documents", metadata=_get_metadata(documents, fed_text)
    )

    return raw_documents


def _get_metadata(documents: list[FedDocument], fed_text: list[FedTextDocument]) -> dict:
    return {
        "per_type": dict(Counter(str(doc.info_type) for doc in documents)),
        "num_documents": len(documents),
        "num_texts": len(fed_text),
        "missing_text": len(documents) - len(fed_text),
    }