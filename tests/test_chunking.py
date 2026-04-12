from __future__ import annotations

from mirage.chunking import SentenceChunker, TokenChunker
from mirage.config import ChunkingConfig
from mirage.schemas import Document


def test_chunker_produces_stable_chunk_ids() -> None:
    document = Document(
        doc_id="doc-001",
        title="Synthetic",
        source="seed://synthetic",
        text="word " * 700,
    )
    chunker = TokenChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))

    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc-001::chunk-0000"
    assert chunks[1].chunk_id == "doc-001::chunk-0001"


def test_sentence_chunker_uses_sentence_aware_windows() -> None:
    document = Document(
        doc_id="doc-002",
        title="Sentence synthetic",
        source="seed://synthetic",
        text=(
            "Alpha beta gamma delta. "
            "Epsilon zeta eta theta. "
            "Iota kappa lambda mu. "
            "Nu xi omicron pi."
        ),
    )
    chunker = SentenceChunker(ChunkingConfig(kind="sentence", chunk_size=12, chunk_overlap=0))

    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc-002::chunk-0000"
    assert chunks[1].chunk_id == "doc-002::chunk-0001"
    assert chunks[0].text.endswith(".")
    assert "Alpha beta gamma delta." in chunks[0].text
    assert "Nu xi omicron pi." in chunks[-1].text
