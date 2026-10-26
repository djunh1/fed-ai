"""Unit tests for the embedding handlers, EmbeddingDispatcher and Query. The model is patched out.

Run: uv run pytest tests/unit/embedding_test.py -v
"""

import datetime
import uuid
from unittest.mock import MagicMock, patch

import pytest

from llm_engineering.application.preprocessing import EmbeddingDispatcher
from llm_engineering.application.preprocessing.embedding_data_handlers import (
    FedEmbeddingHandler,
    QueryEmbeddingHandler,
)
from llm_engineering.domain.chunks import FedChunk
from llm_engineering.domain.embedded_chunks import EmbeddedFedChunk
from llm_engineering.domain.queries import EmbeddedQuery, Query
from llm_engineering.domain.types import FedInfoType, SourceSystem

MODULE = "llm_engineering.application.preprocessing.embedding_data_handlers"


@pytest.fixture
def fake_model():
    """Returns one 3-float vector per input text, like the real model's to_list=True."""
    model = MagicMock(side_effect=lambda texts, to_list: [[float(i)] * 3 for i in range(len(texts))])
    model.model_id, model.embedding_size, model.max_input_length = "fake-model", 3, 256
    with patch(f"{MODULE}.embedding_model", model):
        yield model


def _chunk(content: str = "Rates held.") -> FedChunk:
    return FedChunk(
        content=content,
        document_id=uuid.uuid4(),
        url="https://example.com/minutes.pdf",
        info_type=FedInfoType.MINUTES,
        source=SourceSystem.FED_GOV,
        meeting_date=datetime.datetime(1995, 2, 1),
        title="Minutes",
        metadata={"chunk_size": 500, "chunk_overlap": 50},
    )


def test_embed_batch_calls_model_once_and_pairs_in_order(fake_model) -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]

    embedded = FedEmbeddingHandler().embed_batch(chunks)

    fake_model.assert_called_once_with(["a", "b", "c"], to_list=True)
    assert [e.embedding for e in embedded] == [[0.0] * 3, [1.0] * 3, [2.0] * 3]
    assert [e.content for e in embedded] == ["a", "b", "c"]


def test_fed_handler_copies_chunk_and_merges_metadata(fake_model) -> None:
    chunk = _chunk()

    embedded = FedEmbeddingHandler().embed(chunk)

    assert isinstance(embedded, EmbeddedFedChunk)
    assert embedded.id == chunk.id
    assert embedded.document_id == chunk.document_id
    assert (embedded.info_type, embedded.meeting_date, embedded.title) == (
        chunk.info_type,
        chunk.meeting_date,
        chunk.title,
    )
    assert embedded.metadata == {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_model_id": "fake-model",
        "embedding_size": 3,
        "max_input_length": 256,
    }


def test_query_from_str_needs_only_text() -> None:
    query = Query.from_str("  What did the FOMC decide?\n")

    assert query.content == "What did the FOMC decide?"
    assert query.info_type is None and query.meeting_date is None


def test_query_handler_embeds_question_and_keeps_filters(fake_model) -> None:
    query = Query(content="Minutes in March 2008?", info_type=FedInfoType.MINUTES)

    embedded = QueryEmbeddingHandler().embed(query)

    assert isinstance(embedded, EmbeddedQuery)
    assert embedded.id == query.id
    assert embedded.embedding == [0.0] * 3
    assert embedded.info_type == FedInfoType.MINUTES


def test_dispatcher_embeds_a_list_of_chunks(fake_model) -> None:
    embedded = EmbeddingDispatcher.dispatch([_chunk("a"), _chunk("b")])

    assert [type(e) for e in embedded] == [EmbeddedFedChunk, EmbeddedFedChunk]


def test_dispatcher_unwraps_a_single_query(fake_model) -> None:
    embedded = EmbeddingDispatcher.dispatch(Query.from_str("What happened?"))

    assert isinstance(embedded, EmbeddedQuery)


def test_dispatcher_returns_empty_list_for_empty_input(fake_model) -> None:
    assert EmbeddingDispatcher.dispatch([]) == []
    fake_model.assert_not_called()


def test_dispatcher_rejects_mixed_categories(fake_model) -> None:
    with pytest.raises(AssertionError):
        EmbeddingDispatcher.dispatch([_chunk(), Query.from_str("x")])


def test_replace_content_keeps_id_and_filters() -> None:
    query = Query(content="old", info_type=FedInfoType.STATEMENT, meeting_date=datetime.datetime(2008, 3, 18))

    replaced = query.replace_content("new")

    assert (replaced.id, replaced.content, replaced.info_type, replaced.meeting_date) == (
        query.id,
        "new",
        FedInfoType.STATEMENT,
        datetime.datetime(2008, 3, 18),
    )
