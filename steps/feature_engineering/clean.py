from typing_extensions import Annotated
from zenml import get_step_context, step

#from llm_engineering.application.preprocessing import CleaningDispatcher
from llm_engineering.domain.cleaned_documents import CleanedDocument

@step
def clean_documents():
    pass


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    pass