"""Unit tests for the chunk_and_embed step. The embedding model is patched out.

Run: uv run pytest tests/unit/chunk_and_embed_test.py -v
"""

import datetime
from unittest.mock import MagicMock, patch

from llm_engineering.domain.cleaned_documents import CleanedFedDocument
from llm_engineering.domain.embedded_chunks import EmbeddedFedChunk
from llm_engineering.domain.types import FedInfoType, SourceSystem
from steps.feature_engineering.rag import chunk_and_embed

SENTENCE = "The Committee sold $1.5 billion at 4 3/4 percent."


def _cleaned(content: str) -> CleanedFedDocument:
    return CleanedFedDocument(
        content=content,
        url="https://example.com/minutes.pdf",
        info_type=FedInfoType.MINUTES,
        source=SourceSystem.FED_GOV,
        meeting_date=datetime.datetime(1995, 2, 1),
        title="Minutes",
    )


def _run(documents, embed=lambda texts, to_list: [[0.5] * 3 for _ in texts]):
    model = MagicMock(side_effect=embed)
    model.model_id, model.embedding_size, model.max_input_length = "fake-model", 3, 256
    with (
        patch("llm_engineering.application.preprocessing.embedding_data_handlers.embedding_model", model),
        patch("steps.feature_engineering.rag.get_step_context") as ctx,
    ):
        output = chunk_and_embed.entrypoint(documents)
    return output, ctx.return_value.add_output_metadata.call_args.kwargs, model


def test_chunks_and_embeds_every_document_in_batches_of_10() -> None:
    documents = [_cleaned(" ".join([SENTENCE] * 120)), _cleaned(SENTENCE)]

    output, kwargs, model = _run(documents)

    assert all(isinstance(chunk, EmbeddedFedChunk) for chunk in output)
    assert max(len(call.args[0]) for call in model.call_args_list) <= 10
    assert kwargs["output_name"] == "embedded_documents"
    metadata = kwargs["metadata"]
    assert metadata["num_documents"] == 2
    assert metadata["num_chunks"] == metadata["num_embedded_chunks"] == len(output)
    assert metadata["embedding"]["embedding_model_id"] == "fake-model"


def test_metadata_shows_chunks_lost_when_the_model_fails() -> None:
    """EmbeddingModelSingleton returns [] on error; the counts must expose that."""
    output, kwargs, _ = _run([_cleaned(SENTENCE)], embed=lambda texts, to_list: [])

    assert output == []
    assert kwargs["metadata"]["num_chunks"] == 1
    assert kwargs["metadata"]["num_embedded_chunks"] == 0
    assert kwargs["metadata"]["embedding"] == {}
