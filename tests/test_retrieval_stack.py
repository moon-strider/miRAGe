from __future__ import annotations

from mirage.reranking import rerank_items
from mirage.retrieval import mmr_search, rrf_fuse, sparse_search
from mirage.schemas import RetrievedItem


def _item(chunk_id: str, text: str, score: float = 0.0) -> RetrievedItem:
    return RetrievedItem(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::", 1)[0],
        title=chunk_id,
        source="seed://test",
        text=text,
        score=score,
        order=0,
    )


def test_sparse_search_prefers_matching_chunk() -> None:
    items = [
        _item("doc-001::chunk-0000", "orchards and apples in spring"),
        _item("doc-002::chunk-0000", "bond yields and inflation"),
    ]

    ranked = sparse_search("apple orchards", items, top_k=2)

    assert ranked[0].chunk_id == "doc-001::chunk-0000"


def test_rrf_fuse_combines_rankings() -> None:
    dense = [
        _item("doc-001::chunk-0000", "dense-first", 0.9),
        _item("doc-002::chunk-0000", "dense-second", 0.8),
    ]
    sparse = [
        _item("doc-002::chunk-0000", "sparse-first", 0.7),
        _item("doc-003::chunk-0000", "sparse-second", 0.6),
    ]

    fused = rrf_fuse([dense, sparse], top_k=3, rrf_k=10)

    assert fused[0].chunk_id == "doc-002::chunk-0000"
    assert len(fused) == 3


def test_mmr_search_promotes_diversity() -> None:
    items = [
        _item("doc-001::chunk-0000", "apple orchard"),
        _item("doc-002::chunk-0000", "apple harvest"),
        _item("doc-003::chunk-0000", "bond inflation"),
    ]
    item_vectors = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ]

    ranked = mmr_search([1.0, 0.0], items, item_vectors, top_k=2, lambda_weight=0.2)

    assert ranked[0].chunk_id == "doc-001::chunk-0000"
    assert ranked[1].chunk_id == "doc-003::chunk-0000"


def test_rerank_items_reorders_results(monkeypatch) -> None:
    items = [
        _item("doc-001::chunk-0000", "apples"),
        _item("doc-002::chunk-0000", "bonds"),
    ]

    class FakeEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str], batch_size: int = 64):
            del query, batch_size
            return [0.1 if document == "apples" else 0.9 for document in documents]

    monkeypatch.setattr("mirage.reranking.TextCrossEncoder", FakeEncoder)

    reranked = rerank_items(
        reranker_id="rerank-minilm-l6-v1",
        reranker_kind="cross-encoder",
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        reranker_batch_size=8,
        question="finance",
        items=items,
    )

    assert reranked[0].chunk_id == "doc-002::chunk-0000"


def test_rerank_items_supports_multiple_registered_models(monkeypatch) -> None:
    seen_models: list[str] = []

    class FakeEncoder:
        def __init__(self, model_name: str) -> None:
            seen_models.append(model_name)
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str], batch_size: int = 64):
            del query, documents, batch_size
            return [1.0]

    monkeypatch.setattr("mirage.reranking.TextCrossEncoder", FakeEncoder)

    for reranker_id, model_name in [
        ("rerank-minilm-l12-v1", "Xenova/ms-marco-MiniLM-L-12-v2"),
        ("rerank-jina-tiny-v1", "jinaai/jina-reranker-v1-tiny-en"),
        ("rerank-jina-turbo-v1", "jinaai/jina-reranker-v1-turbo-en"),
    ]:
        rerank_items(
            reranker_id=reranker_id,
            reranker_kind="cross-encoder",
            reranker_model=model_name,
            reranker_batch_size=16,
            question="finance",
            items=[_item("doc-001::chunk-0000", "alpha")],
        )

    assert seen_models == [
        "Xenova/ms-marco-MiniLM-L-12-v2",
        "jinaai/jina-reranker-v1-tiny-en",
        "jinaai/jina-reranker-v1-turbo-en",
    ]
