from abc import ABC
import datetime

from .base import VectorBaseDocument
from .types import DataCategory, FedInfoType, SourceSystem


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    url: str
    

class CleanedFedDocument(CleanedDocument):
    info_type: FedInfoType
    source: SourceSystem
    meeting_date: datetime.datetime
    title: str

    class Config:
        name = "cleaned_fed_docs"
        category = DataCategory.FED_TEXT
        use_vector_index = False