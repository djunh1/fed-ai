from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.base import NoSQLBaseDocument

from llm_engineering.domain.cleaned_documents import (
    CleanedDocument,
    CleanedFedDocument,
)

from llm_engineering.domain.documents import (
    FedTextDocument,
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=NoSQLBaseDocument)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)

class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass

class FedCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: FedTextDocument) -> CleanedFedDocument:
        return CleanedFedDocument(
            id=data_model.id,
            content=clean_text(data_model.text),
            url=data_model.url,
            info_type=data_model.info_type,
            source=data_model.source,
            meeting_date=data_model.meeting_date,
            title=data_model.title,
        )