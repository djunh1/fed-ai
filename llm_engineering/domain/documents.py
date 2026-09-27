import datetime

from abc import ABC
from typing import Optional
from dataclasses import dataclass

from pydantic import UUID4, Field

from llm_engineering.application.api.models import FedInfoType, SourceSystem
from .base import NoSQLBaseDocument


class FedDocument(NoSQLBaseDocument):
    info_type: str

    class Settings:
        name = "fed"

    @property
    def get_meeting_type(self):
        return f"{self.meeting_type}"


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")
