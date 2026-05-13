from __future__ import annotations

from types import SimpleNamespace

import httpx
from openai import APIStatusError

from mirage.chunking import EmbeddingBoundaryChunker, SentenceChunker, TokenChunker
from mirage.config import ChunkingConfig
from mirage.llm_chunking import BoundaryDecision, LlmSemanticChunker, OpenRouterBoundaryPlanner
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


def test_embedding_boundary_chunker_groups_neighboring_sentences_by_similarity() -> None:
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

    chunker = EmbeddingBoundaryChunker(
        ChunkingConfig(
            kind="embedding-boundary",
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


def test_embedding_boundary_chunker_honors_min_sentences_before_split() -> None:
    document = Document(
        doc_id="doc-004",
        title="Semantic min sentences",
        source="seed://synthetic",
        text="Alpha one. Beta two. Gamma three.",
    )

    def embedder(texts: list[str]) -> list[list[float]]:
        mapping = {
            "Alpha one.": [1.0, 0.0],
            "Beta two.": [0.0, 1.0],
            "Gamma three.": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]

    chunker = EmbeddingBoundaryChunker(
        ChunkingConfig(
            kind="embedding-boundary",
            chunk_size=128,
            chunk_overlap=0,
            chunking_model_id="emb-test",
            semantic_similarity_threshold=0.9,
            semantic_min_sentences_per_chunk=2,
        ),
        embedder=embedder,
    )

    chunks = chunker.chunk_documents([document])

    assert len(chunks) == 2
    assert "Alpha one. Beta two." in chunks[0].text
    assert chunks[1].text == "Gamma three."


def test_embedding_boundary_chunker_respects_chunk_size_cap() -> None:
    document = Document(
        doc_id="doc-005",
        title="Semantic chunk size",
        source="seed://synthetic",
        text="One two three four. Five six seven eight. Nine ten eleven twelve.",
    )

    def embedder(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    chunker = EmbeddingBoundaryChunker(
        ChunkingConfig(
            kind="embedding-boundary",
            chunk_size=6,
            chunk_overlap=0,
            chunking_model_id="emb-test",
            semantic_similarity_threshold=0.1,
            semantic_min_sentences_per_chunk=1,
        ),
        embedder=embedder,
    )

    chunks = chunker.chunk_documents([document])

    assert len(chunks) == 3


def test_llm_semantic_chunker_uses_planned_unit_boundaries() -> None:
    document = Document(
        doc_id="doc-006",
        title="LLM semantic",
        source="seed://synthetic",
        text="Apples need water.\n\nApple trees bloom.\n\nMarkets price risk.\n\nBond yields move.",
    )
    calls = []

    def planner(units, max_tokens):
        calls.append([unit.unit_id for unit in units])
        return BoundaryDecision(boundary_unit_id="u_0002", reason="topic shift", confidence=0.9)

    chunker = LlmSemanticChunker(
        ChunkingConfig(kind="semantic", chunk_size=128, chunk_overlap=0, chunking_model_id="gen-test"),
        boundary_planner=planner,
        model="provider/test",
    )

    chunks = chunker.chunk_document(document)

    assert calls == [["u_0000", "u_0001", "u_0002", "u_0003"]]
    assert len(chunks) == 2
    assert chunks[0].text == "Apples need water.\n\nApple trees bloom."
    assert chunks[1].text == "Markets price risk.\n\nBond yields move."
    assert chunks[0].metadata["chunking_strategy"] == "llm-semantic"
    assert chunks[0].metadata["unit_ids"] == ["u_0000", "u_0001"]


def test_llm_semantic_chunker_batches_document_plans(tmp_path) -> None:
    documents = [
        Document(doc_id="doc-a", title="A", source="seed://a", text="Apples need water.\n\nApple trees bloom."),
        Document(doc_id="doc-b", title="B", source="seed://b", text="Markets price risk.\n\nBond yields move."),
    ]
    batch_calls = []

    def batch_planner(requests):
        batch_calls.append([request_id for request_id, _, _ in requests])
        return {
            request_id: BoundaryDecision(boundary_unit_id=None, reason="same topic", confidence=1.0)
            for request_id, _, _ in requests
        }

    chunker = LlmSemanticChunker(
        ChunkingConfig(kind="semantic", chunk_size=128, chunk_overlap=0, chunking_model_id="gen-test"),
        boundary_planner=lambda units, max_tokens: BoundaryDecision(boundary_unit_id=None, reason="unused", confidence=1.0),
        batch_boundary_planner=batch_planner,
        batch_size=4,
        cache_dir=tmp_path,
        model="provider/test",
    )

    chunks = chunker.chunk_documents(documents)

    assert batch_calls == [["r_0000", "r_0001"]]
    assert [chunk.doc_id for chunk in chunks] == ["doc-a", "doc-b"]
    assert len(list(tmp_path.glob("**/*.json"))) == 2


def test_llm_semantic_chunker_reuses_cached_plan(tmp_path) -> None:
    document = Document(
        doc_id="doc-007",
        title="LLM cache",
        source="seed://synthetic",
        text="First topic.\n\nSecond topic.\n\nThird topic.",
    )
    calls = 0

    def planner(units, max_tokens):
        nonlocal calls
        calls += 1
        return BoundaryDecision(boundary_unit_id="u_0001", reason="topic shift", confidence=0.8)

    chunker = LlmSemanticChunker(
        ChunkingConfig(kind="semantic", chunk_size=128, chunk_overlap=0, chunking_model_id="gen-test"),
        boundary_planner=planner,
        cache_dir=tmp_path,
        model="provider/test",
    )

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert calls == 1
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_llm_semantic_chunking_preflight_counts_full_documents(tmp_path) -> None:
    documents = [
        Document(doc_id="a", title="A", source="seed://a", text="One.\n\nTwo."),
        Document(doc_id="b", title="B", source="seed://b", text="Three.\n\nFour.\n\nFive."),
    ]
    chunker = LlmSemanticChunker(
        ChunkingConfig(kind="semantic", chunk_size=128, chunk_overlap=0, chunking_model_id="gen-test"),
        boundary_planner=lambda units, max_tokens: BoundaryDecision(boundary_unit_id=None, reason="same", confidence=1.0),
        cache_dir=tmp_path,
        model="provider/test",
    )

    preflight = chunker.preflight(documents)

    assert preflight.documents == 2
    assert preflight.units == 5
    assert preflight.cached_plans == 0
    assert preflight.missing_plans == 2
    assert preflight.estimated_llm_calls == 2


def test_openrouter_boundary_planner_retries_transient_statuses_with_constant_backoff() -> None:
    calls = 0
    sleeps = []

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls <= 2:
                status = 429 if calls == 1 else 503
                request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
                response = httpx.Response(status, request=request)
                raise APIStatusError("transient", response=response, body=None)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"boundary_unit_id": null, "reason": "same topic", "confidence": 1.0}'
                        )
                    )
                ]
            )

    planner = OpenRouterBoundaryPlanner.__new__(OpenRouterBoundaryPlanner)
    planner._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    planner._model = "test/model"
    planner._temperature = 0.0
    planner._rate_limit_backoff_seconds = 1.0
    planner._sleep = sleeps.append

    result = planner._create_completion([])

    assert calls == 3
    assert sleeps == [1.0, 1.0]
    assert result.choices[0].message.content.startswith("{")
