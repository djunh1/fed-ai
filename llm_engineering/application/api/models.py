import datetime

from dataclasses import dataclass

from llm_engineering.domain.types import SourceSystem, FedInfoType


@dataclass(frozen=True, slots=True)
class FedSourceDocument:
    source: SourceSystem
    info_type: FedInfoType
    meeting_date: datetime.date
    title: str
    url: str
    text_url: str | None = None
    source_id: str | None = None
    media_type: str = "application/pdf"
    page_count: int | None = None
