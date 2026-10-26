"""Unit tests for chunk_text and FedChunkingHandler. No database.

Run: uv run pytest tests/unit/chunking_test.py -v
"""

import datetime

from llm_engineering.application.preprocessing.chunking_data_handlers import FedChunkingHandler
from llm_engineering.application.preprocessing.operations import chunk_text
from llm_engineering.domain.chunks import FedChunk
from llm_engineering.domain.cleaned_documents import CleanedFedDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem

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


def test_chunk_text_respects_size_and_keeps_case_and_symbols() -> None:
    chunks = chunk_text(" ".join([SENTENCE] * 40), chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert chunks[0].startswith(SENTENCE)


def test_chunk_text_prefers_line_breaks() -> None:
    lines = [f"Line {i} " + "x" * 80 for i in range(10)]

    chunks = chunk_text("\n".join(lines), chunk_size=200, chunk_overlap=0)

    assert all(chunk.startswith("Line ") for chunk in chunks)


def test_handler_copies_metadata_and_links_back() -> None:
    doc = _cleaned(" ".join([SENTENCE] * 40))

    chunks = FedChunkingHandler().chunk(doc)

    assert all(isinstance(chunk, FedChunk) for chunk in chunks)
    assert {chunk.document_id for chunk in chunks} == {doc.id}
    assert {(chunk.info_type, chunk.meeting_date, chunk.url) for chunk in chunks} == {
        (doc.info_type, doc.meeting_date, doc.url)
    }
    assert chunks[0].metadata == {"chunk_size": 500, "chunk_overlap": 50}


def test_same_text_in_two_documents_gets_different_ids() -> None:
    """A content-only hash would collide and Qdrant would overwrite one meeting's chunk."""
    first = FedChunkingHandler().chunk(_cleaned(SENTENCE))
    second = FedChunkingHandler().chunk(_cleaned(SENTENCE))

    assert first[0].content == second[0].content
    assert first[0].id != second[0].id


def test_ids_are_stable_across_runs() -> None:
    doc = _cleaned(SENTENCE)

    assert FedChunkingHandler().chunk(doc)[0].id == FedChunkingHandler().chunk(doc)[0].id
