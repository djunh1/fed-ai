from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.application.preprocessing import ChunkingDispatcher, EmbeddingDispatcher


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    num_chunks = 0
    embedded_chunks = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        num_chunks += len(chunks)

        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_documents",
        metadata={
            "num_documents": len(cleaned_documents),
            "num_chunks": num_chunks,
            "num_embedded_chunks": len(embedded_chunks),
            "embedding": embedded_chunks[0].metadata if embedded_chunks else {},
        },
    )

    return embedded_chunks
