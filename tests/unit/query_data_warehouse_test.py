"""Unit tests for the query_data_warehouse step. No database: bulk_find is patched.

Run: uv run pytest tests/unit/query_data_warehouse_test.py -v
"""

import datetime
from unittest.mock import patch

import pytest

from llm_engineering.domain.documents import FedDocument, FedTextDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem
from steps.feature_engineering import query_data_warehouse as step

MODULE = "steps.feature_engineering.query_data_warehouse"


def _document(url: str, info_type: FedInfoType = FedInfoType.MINUTES) -> FedDocument:
    return FedDocument(
        source=SourceSystem.FED_GOV,
        info_type=info_type,
        meeting_date=datetime.datetime(1995, 2, 1),
        title=f"Title {url}",
        url=url,
    )


def _text(url: str) -> FedTextDocument:
    return FedTextDocument(url=url, text=f"text {url}", char_count=9)


@pytest.fixture
def run():
    """Runs the step with the given Mongo rows; returns (output, filters, metadata)."""

    def _run(documents, texts, fed_doc_types=("minutes",)):
        with (
            patch.object(FedDocument, "bulk_find", return_value=documents) as doc_find,
            patch.object(FedTextDocument, "bulk_find", return_value=texts) as text_find,
            patch(f"{MODULE}.get_step_context") as ctx,
        ):
            output = step.entrypoint(list(fed_doc_types))
        filters = (doc_find.call_args.kwargs, text_find.call_args.kwargs)
        metadata = ctx.return_value.add_output_metadata.call_args.kwargs
        return output, filters, metadata

    return _run


def test_queries_by_info_type_then_by_document_urls(run) -> None:
    documents = [_document("a"), _document("b")]

    _, (doc_filter, text_filter), _ = run(documents, [], fed_doc_types=["minutes", "statement"])

    assert doc_filter == {"info_type": {"$in": ["minutes", "statement"]}}
    assert text_filter == {"url": {"$in": ["a", "b"]}}


def test_returns_one_list_of_text_rows_with_metadata_joined(run) -> None:
    documents = [_document("a"), _document("b", FedInfoType.STATEMENT)]
    texts = [_text("b"), _text("a")]

    output, _, _ = run(documents, texts)

    assert [type(row) for row in output] == [FedTextDocument, FedTextDocument]
    by_url = {row.url: row for row in output}
    assert by_url["b"].info_type == FedInfoType.STATEMENT
    assert by_url["a"].source == SourceSystem.FED_GOV
    assert by_url["a"].meeting_date == datetime.datetime(1995, 2, 1)
    assert by_url["a"].title == "Title a"
    assert by_url["a"].text == "text a"


def test_join_does_not_mutate_the_queried_rows(run) -> None:
    texts = [_text("a")]

    run([_document("a")], texts)

    assert texts[0].info_type is None


def test_metadata_counts_types_and_missing_text(run) -> None:
    documents = [_document("a"), _document("b"), _document("c", FedInfoType.STATEMENT)]

    _, _, metadata = run(documents, [_text("a")])

    assert metadata["output_name"] == "raw_documents"
    assert metadata["metadata"] == {
        "per_type": {"minutes": 2, "statement": 1},
        "num_documents": 3,
        "num_texts": 1,
        "missing_text": 2,
    }


def test_no_documents_returns_empty_list(run) -> None:
    output, _, metadata = run([], [])

    assert output == []
    assert metadata["metadata"]["missing_text"] == 0
