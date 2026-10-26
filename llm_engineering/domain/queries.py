import datetime

from pydantic import Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory, FedInfoType


class Query(VectorBaseDocument):
    content: str
    info_type: FedInfoType | None = None
    meeting_date: datetime.datetime | None = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        category = DataCategory.QUERIES

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            info_type=self.info_type,
            meeting_date=self.meeting_date,
            metadata=self.metadata,
        )


class EmbeddedQuery(Query):
    embedding: list[float]

    class Config:
        category = DataCategory.QUERIES
