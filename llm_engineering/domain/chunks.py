from abc import ABC
import datetime

from pydantic import UUID4, Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory, FedInfoType, SourceSystem


class Chunk(VectorBaseDocument, ABC):
    content: str
    document_id: UUID4
    url: str
    metadata: dict = Field(default_factory=dict)
    

class FedChunk(Chunk):
    info_type: FedInfoType
    source: SourceSystem
    meeting_date: datetime.datetime
    title: str

    class Config:
        category = DataCategory.FED_TEXT

