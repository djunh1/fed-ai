from abc import ABC, abstractmethod
import datetime

from pydantic import UUID4, Field

from llm_engineering.domain.types import DataCategory, FedInfoType, SourceSystem

from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    document_id: UUID4
    url: str
    metadata: dict = Field(default_factory=dict)

    @abstractmethod
    def _context_lines(self) -> str:
        ...

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Url: {chunk.url}
            {chunk._context_lines()}
            Content: {chunk.content}\n
            """

        return context


class EmbeddedFedChunk(EmbeddedChunk):
    info_type: FedInfoType
    source: SourceSystem
    meeting_date: datetime.datetime
    title: str

    def _context_lines(self) -> str:
        title = f"Title: {self.title}\n"
        info_type = f"Info Type: {self.info_type}\n"
        meeting_date = f"Meeting Date: {self.meeting_date:%Y-%m-%d}"

        return "".join([title, info_type, meeting_date])

    class Config:
        name = "embedded_fed_docs"
        category = DataCategory.FED_TEXT
        use_vector_index = True

