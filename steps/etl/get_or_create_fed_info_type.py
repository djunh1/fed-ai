from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step


from llm_engineering.domain.documents import FedDocument


@step
def get_or_create_fed_info_type(
    fed_info_type: str,
) -> Annotated[FedDocument, "info_type"]:
    logger.info(f"Getting or creating fed info type: {fed_info_type}")
    info_type = FedDocument.get_or_create(fed_info_type)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="info_type", metadata=_get_metadata(fed_info_type, info_type)
    )

    return info_type


def _get_metadata(fed_info_type: str, fed: FedDocument) -> dict:
    return {
        "query": {
            "fed_info_type": fed_info_type,
        },
        "retrieved": {
            "type_type_id": str(fed.id),
            "first_name": fed.info_type,
        },
    }
