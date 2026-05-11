from __future__ import annotations

from mirage.faiss_store import build_faiss_index, load_faiss_index, save_faiss_index, search_faiss_index
from mirage.reranking import rerank_items
from mirage.retrieval import rrf_fuse, sparse_search
from mirage.schemas import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::", 1)[0],
        title=chunk_id,
        source="seed://test",
        text=text,
        order=0,
        token_count=4,
    )


def test_build_and_search_faiss_flat_index(tmp_path) -> None:
    chunks = [
        _chunk("doc-001::chunk-0000", "apples and orchards"),
        _chunk("doc-002::chunk-0000", "bonds and inflation"),
    ]
    index = build_faiss_index("flat", [[1.0, 0.0], [0.0, 1.0]])

    results = search_faiss_index(index=index, query_vector=[0.9, 0.1], items=chunks, top_k=2)

    assert results[0].chunk_id == "doc-001::chunk-0000"
    assert len(results) == 2

    sparse = sparse_search(
        "inflation bonds",
        [
            results[0].model_copy(update={"text": "apples and orchards"}),
            results[1].model_copy(update={"text": "bonds and inflation"}),
        ],
        top_k=2,
    )
    fused = rrf_fuse([results, sparse], top_k=2, rrf_k=10)
    assert len(fused) == 2

    path = tmp_path / "index.faiss"
    save_faiss_index(index, path)
    restored = load_faiss_index(path)
    restored_results = search_faiss_index(index=restored, query_vector=[0.1, 0.9], items=chunks, top_k=2)
    assert restored_results[0].chunk_id == "doc-002::chunk-0000"


def test_build_and_search_faiss_ivfflat_index(tmp_path) -> None:
    chunks = [
        _chunk(f"doc-{idx:03d}::chunk-0000", f"topic {idx}")
        for idx in range(16)
    ]
    vectors = [[1.0, 0.0] if idx < 8 else [0.0, 1.0] for idx in range(16)]
    index = build_faiss_index("ivfflat", vectors, nlist=2)

    results = search_faiss_index(index=index, query_vector=[0.95, 0.05], items=chunks, top_k=4)

    assert len(results) == 4
    assert all(item.doc_id.startswith("doc-00") for item in results)

    path = tmp_path / "ivfflat.faiss"
    save_faiss_index(index, path)
    restored = load_faiss_index(path)
    restored_results = search_faiss_index(index=restored, query_vector=[0.05, 0.95], items=chunks, top_k=4)
    assert len(restored_results) == 4


def test_build_and_search_faiss_ivfpq_index(tmp_path) -> None:
    chunks = [
        _chunk(f"doc-{idx:03d}::chunk-0000", f"topic {idx}")
        for idx in range(32)
    ]
    vectors = []
    for idx in range(32):
        if idx < 16:
            vectors.append([1.0, 0.0, 0.0, 0.0])
        else:
            vectors.append([0.0, 1.0, 0.0, 0.0])
    index = build_faiss_index("ivfpq", vectors, nlist=2, m=2, bits=4)

    results = search_faiss_index(index=index, query_vector=[0.0, 1.0, 0.0, 0.0], items=chunks, top_k=5)

    assert len(results) == 5
    path = tmp_path / "ivfpq.faiss"
    save_faiss_index(index, path)
    restored = load_faiss_index(path)
    restored_results = search_faiss_index(index=restored, query_vector=[1.0, 0.0, 0.0, 0.0], items=chunks, top_k=5)
    assert len(restored_results) == 5


def test_faiss_results_can_be_reranked(monkeypatch) -> None:
    chunks = [
        _chunk("doc-001::chunk-0000", "apples"),
        _chunk("doc-002::chunk-0000", "bonds"),
    ]
    index = build_faiss_index("flat", [[1.0, 0.0], [0.0, 1.0]])
    dense = search_faiss_index(index=index, query_vector=[0.5, 0.5], items=chunks, top_k=2)

    class FakeEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str], batch_size: int = 64):
            del query, batch_size
            return [0.2 if document == "apples" else 0.8 for document in documents]

    monkeypatch.setattr("mirage.reranking.TextCrossEncoder", FakeEncoder)
    reranked = rerank_items(
        reranker_id="rerank-minilm-l6-v1",
        reranker_kind="cross-encoder",
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        reranker_batch_size=8,
        question="finance",
        items=dense,
    )

    assert reranked[0].chunk_id == "doc-002::chunk-0000"
