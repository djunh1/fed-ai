import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_engineering.domain.chunks import Chunk, FedChunk
from llm_engineering.domain.cleaned_documents import (
    CleanedFedDocument,
)

from .operations import chunk_text

CleanedFedDocumentT = TypeVar("CleanedFedDocumentT", bound=CleanedFedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)

class ChunkingDataHandler(ABC, Generic[CleanedFedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @abstractmethod
    def chunk(self, data_model: CleanedFedDocumentT) -> list[ChunkT]:
        pass

class FedChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    def chunk(self, data_model: CleanedFedDocument) -> list[FedChunk]:
        data_models_list = []


        chunks = chunk_text(
            data_model.content, 
            chunk_size=self.metadata["chunk_size"], 
            chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(f"{data_model.id}{chunk}".encode()).hexdigest()
            model = FedChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                document_id=data_model.id,
                url=data_model.url,
                info_type=data_model.info_type,
                source=data_model.source,
                meeting_date=data_model.meeting_date,
                title=data_model.title,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list