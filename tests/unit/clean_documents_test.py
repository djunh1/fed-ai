"""Unit tests for the clean_documents step. No database.

Run: uv run pytest tests/unit/clean_documents_test.py -v
"""

import datetime
from unittest.mock import patch

from llm_engineering.domain.cleaned_documents import CleanedFedDocument
from llm_engineering.domain.documents import FedTextDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem
from steps.feature_engineering.clean import clean_documents


def _joined_text(info_type: FedInfoType) -> FedTextDocument:
    return FedTextDocument(
        url=f"https://example.com/{info_type}",
        text="Rates  held.",
        char_count=12,
        info_type=info_type,
        source=SourceSystem.FED_GOV,
        meeting_date=datetime.datetime(1995, 2, 1),
        title="Title",
    )


def test_cleans_every_document_and_reports_metadata() -> None:
    raw = [_joined_text(FedInfoType.MINUTES), _joined_text(FedInfoType.MINUTES), _joined_text(FedInfoType.STATEMENT)]

    with patch("steps.feature_engineering.clean.get_step_context") as ctx:
        cleaned = clean_documents.entrypoint(raw)

    assert [type(doc) for doc in cleaned] == [CleanedFedDocument] * 3
    assert cleaned[0].content == "Rates held."

    kwargs = ctx.return_value.add_output_metadata.call_args.kwargs
    assert kwargs["output_name"] == "cleaned_documents"
    assert kwargs["metadata"]["num_documents"] == 3
    assert kwargs["metadata"]["fed_text"]["num_documents"] == 3
    assert sorted(kwargs["metadata"]["fed_text"]["info_type"]) == ["minutes", "statement"]
