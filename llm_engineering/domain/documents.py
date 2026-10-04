import datetime
from abc import ABC

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument
from .types import FedInfoType, SourceSystem


def to_bson_datetime(value: datetime.date) -> datetime.datetime:
    """BSON has no date type, only datetime.

    pymongo raises InvalidDocument on a bare `datetime.date`, so meeting dates
    are widened to midnight before they reach Mongo. Kept here rather than in
    the mapper so every writer goes through the same conversion.
    """
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.combine(value, datetime.time.min)


class FedDocument(NoSQLBaseDocument):
    """One FOMC document's metadata. Text lives in FedTextDocument.

    Split from the text because NoSQLBaseDocument.save() is a bare insert_one
    with no upsert -- keeping them apart means both the manifest step and the
    extraction step only ever insert, which is all the base class supports.
    """

    source: SourceSystem
    info_type: FedInfoType
    meeting_date: datetime.datetime
    title: str
    url: str
    media_type: str = "application/pdf"
    text_url: str | None = None
    source_id: str | None = None
    page_count: int | None = None

    class Settings:
        name = "fed_documents"


class FedTextDocument(NoSQLBaseDocument):
    """Extracted plain text for one document, keyed by its source URL.

    `page_chars` and `failed_pages` come straight from ExtractedText: a PDF
    that yields an empty string per page is a scan with no OCR layer, and
    that is indistinguishable from a page that threw unless both are recorded.
    """

    url: str
    text: str
    char_count: int
    page_chars: list[int] = Field(default_factory=list)
    failed_pages: list[int] = Field(default_factory=list)
    extracted_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    class Settings:
        name = "fed_text"


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")
