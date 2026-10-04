"""Unit tests for the Mongo document models.

No database. These exercise the pure mapping layer -- pydantic validation and
the to_mongo/from_mongo round trip -- which is where the BSON date problem
and the alias/_id handling live.

Run: uv run pytest tests/unit/documents_test.py -v
"""

import datetime

import pytest

from llm_engineering.domain.documents import FedDocument, FedTextDocument, to_bson_datetime
from llm_engineering.domain.types import FedInfoType, SourceSystem

MEETING = datetime.date(1995, 2, 1)


def _document(**overrides) -> FedDocument:
    fields = {
        "source": SourceSystem.FED_GOV,
        "info_type": FedInfoType.MINUTES,
        "meeting_date": to_bson_datetime(MEETING),
        "title": "Minutes",
        "url": "https://www.federalreserve.gov/fomc/MINUTES/1995/19950201min.htm",
        "media_type": "text/html",
    }
    return FedDocument(**{**fields, **overrides})


# --------------------------------------------------------------------------
# to_bson_datetime
# --------------------------------------------------------------------------


def test_date_is_widened_to_datetime() -> None:
    """BSON has no date type. pymongo raises InvalidDocument on a bare date,
    so this conversion is what stands between the ETL and a hard failure on
    the very first insert."""
    result = to_bson_datetime(MEETING)

    assert isinstance(result, datetime.datetime)
    assert result == datetime.datetime(1995, 2, 1, 0, 0)


def test_datetime_passes_through_unchanged() -> None:
    """datetime is a subclass of date, so a naive isinstance(date) check would
    silently truncate the time component."""
    moment = datetime.datetime(1995, 2, 1, 14, 30)

    assert to_bson_datetime(moment) == moment


# --------------------------------------------------------------------------
# FedDocument
# --------------------------------------------------------------------------


def test_round_trips_through_mongo_form() -> None:
    document = _document()

    assert FedDocument.from_mongo(document.to_mongo()) == document


def test_id_becomes_underscore_id() -> None:
    """Equality is by id, so losing it in translation would make every
    re-read a different document."""
    document = _document()
    stored = document.to_mongo()

    assert "_id" in stored
    assert "id" not in stored
    assert stored["_id"] == str(document.id)


def test_meeting_date_survives_as_a_datetime() -> None:
    stored = _document().to_mongo()

    assert isinstance(stored["meeting_date"], datetime.datetime)


def test_every_source_field_is_carried() -> None:
    """The mapper in build_fed_manifest fills all nine; a field missing from
    the model would be dropped silently rather than raising."""
    expected = {
        "source",
        "info_type",
        "meeting_date",
        "title",
        "url",
        "media_type",
        "text_url",
        "source_id",
        "page_count",
    }

    assert expected <= set(_document().to_mongo())


def test_enums_are_accepted_as_strings() -> None:
    """Values come back off the wire as plain strings, not enum members."""
    document = _document(source="fed_gov", info_type="transcript")

    assert document.info_type is FedInfoType.TRANSCRIPT
    assert document.source is SourceSystem.FED_GOV


def test_unknown_info_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        _document(info_type="not_a_real_type")


def test_optional_fields_default_to_none() -> None:
    document = _document()

    assert (document.text_url, document.source_id, document.page_count) == (None, None, None)


def test_collection_name() -> None:
    assert FedDocument.get_collection_name() == "fed_documents"


# --------------------------------------------------------------------------
# FedTextDocument
# --------------------------------------------------------------------------


def test_text_document_round_trips() -> None:
    text = FedTextDocument(
        url="https://example.org/a.pdf",
        text="hello",
        char_count=5,
        page_chars=[3, 2],
        failed_pages=[1],
    )

    assert FedTextDocument.from_mongo(text.to_mongo()) == text


def test_text_document_stamps_itself() -> None:
    text = FedTextDocument(url="https://example.org/a.pdf", text="hello", char_count=5)

    assert isinstance(text.extracted_at, datetime.datetime)


def test_text_document_page_lists_default_empty() -> None:
    """HTML has no pages. Defaulting these to a shared mutable list would
    leak page counts between documents."""
    first = FedTextDocument(url="https://a", text="x", char_count=1)
    second = FedTextDocument(url="https://b", text="y", char_count=1)

    first.page_chars.append(99)

    assert second.page_chars == []


def test_text_lives_in_its_own_collection() -> None:
    """Separate from fed_documents so both steps only ever insert --
    NoSQLBaseDocument.save() is insert_one with no upsert path."""
    assert FedTextDocument.get_collection_name() == "fed_text"
    assert FedTextDocument.get_collection_name() != FedDocument.get_collection_name()
