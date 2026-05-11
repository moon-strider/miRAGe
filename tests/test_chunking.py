from __future__ import annotations
from __future__ import annotations
from mirage.chunking import SemanticChunker, SentenceChunker, TokenChunker
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


def test_semantic_chunker_groups_neighboring_sentences_by_similarity() -> None:
    document = Document(
        doc_id="doc-003",
        title="Semantic synthetic",
        source="seed://synthetic",
        text=(
            "Apple orchards need water. "
            "Apple trees bloom in spring. "
            "Stock markets price future cash flows. "
            "Bond yields react to inflation."
        ),
    )

    def embedder(texts: list[str]) -> list[list[float]]:
        mapping = {
            "Apple orchards need water.": [1.0, 0.0],
            "Apple trees bloom in spring.": [0.95, 0.05],
            "Stock markets price future cash flows.": [0.0, 1.0],
            "Bond yields react to inflation.": [0.05, 0.95],
        }
        return [mapping[text] for text in texts]

    chunker = SemanticChunker(
        ChunkingConfig(
            kind="semantic",
            chunk_size=128,
            chunk_overlap=0,
            chunking_model_id="emb-test",
            semantic_similarity_threshold=0.8,
            semantic_min_sentences_per_chunk=1,
        ),
        embedder=embedder,
    )

    chunks = chunker.chunk_documents([document])

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "doc-003::chunk-0000"
    assert chunks[1].chunk_id == "doc-003::chunk-0001"
    assert "Apple orchards need water." in chunks[0].text
    assert "Apple trees bloom in spring." in chunks[0].text
    assert "Stock markets price future cash flows." in chunks[1].text
    assert "Bond yields react to inflation." in chunks[1].text
