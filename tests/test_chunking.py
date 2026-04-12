from mirage.chunking import TokenChunker
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
