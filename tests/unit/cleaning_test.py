"""Unit tests for clean_text, FedCleaningHandler and CleaningDispatcher. No database.

Run: uv run pytest tests/unit/cleaning_test.py -v
"""

import datetime

import pytest
from pydantic import ValidationError

from llm_engineering.application.preprocessing import CleaningDispatcher
from llm_engineering.application.preprocessing.cleaning_data_handlers import FedCleaningHandler
from llm_engineering.application.preprocessing.operations import clean_text
from llm_engineering.domain.cleaned_documents import CleanedFedDocument
from llm_engineering.domain.documents import FedDocument, FedTextDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem

# --------------------------------------------------------------------------
# clean_text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\n\n\nb", "a\nb"),
        ("a  \n  b", "a\nb"),
        ("a\n\n\x0cb", "a\nb"),
        ("a \t\xa0 b", "a b"),
        ("  a b  \n", "a b"),
        ("", ""),
    ],
    ids=["blank-lines", "spaces-around-newline", "form-feed", "mixed-spaces", "strip", "empty"],
)
def test_clean_text_normalizes_whitespace(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_keeps_fed_symbols() -> None:
    """The book's regex dropped these; Fed numbers depend on them."""
    text = "from 4 3/4 percent to 5 1/4 percent; (FOMC) sold $1.5 billion, 2-1/2% in Q1-1995."

    assert clean_text(text) == text


# --------------------------------------------------------------------------
# FedCleaningHandler
# --------------------------------------------------------------------------


def _joined_text(**overrides) -> FedTextDocument:
    fields = {
        "url": "https://example.com/minutes.pdf",
        "text": "Minutes  of\n\nthe meeting",
        "char_count": 24,
        "info_type": FedInfoType.MINUTES,
        "source": SourceSystem.FRASER,
        "meeting_date": datetime.datetime(1995, 2, 1),
        "title": "Minutes",
    }
    return FedTextDocument(**{**fields, **overrides})


def test_handler_cleans_content_and_copies_metadata() -> None:
    raw = _joined_text()

    cleaned = FedCleaningHandler().clean(raw)

    assert isinstance(cleaned, CleanedFedDocument)
    assert cleaned.id == raw.id
    assert cleaned.content == "Minutes of\nthe meeting"
    assert (cleaned.url, cleaned.info_type, cleaned.source, cleaned.title) == (
        raw.url,
        FedInfoType.MINUTES,
        SourceSystem.FRASER,
        "Minutes",
    )
    assert cleaned.meeting_date == datetime.datetime(1995, 2, 1)


def test_handler_rejects_rows_that_skipped_the_join() -> None:
    """Joined fields are optional on FedTextDocument but required once cleaned."""
    raw = FedTextDocument(url="u", text="t", char_count=1)

    with pytest.raises(ValidationError):
        FedCleaningHandler().clean(raw)


# --------------------------------------------------------------------------
# CleaningDispatcher
# --------------------------------------------------------------------------


def test_dispatcher_routes_fed_text_to_fed_handler() -> None:
    assert isinstance(CleaningDispatcher.dispatch(_joined_text()), CleanedFedDocument)


def test_dispatcher_rejects_metadata_rows() -> None:
    document = FedDocument(
        source=SourceSystem.FED_GOV,
        info_type=FedInfoType.MINUTES,
        meeting_date=datetime.datetime(1995, 2, 1),
        title="Minutes",
        url="u",
    )

    with pytest.raises(ValueError):
        CleaningDispatcher.dispatch(document)
