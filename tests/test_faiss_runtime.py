from __future__ import annotations

from mirage.config import load_experiment_specs
from mirage.ingest import ingest_spec
from mirage.pipeline import retrieve
from mirage.schemas import Document


def test_faiss_ivfflat_ingest_and_retrieve(monkeypatch, tmp_path) -> None:
    spec = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "generation_model_id": "gen-llama-3.1-8b",
            "store_backend_id": "faiss-local",
            "store_index_variant_id": "idx-faiss-ivfflat-v1",
        },
    )[0].model_copy(update={"artifacts_dir": str(tmp_path / "artifacts"), "store_index_nlist": 2})

    documents = [
        Document(doc_id="doc-001", title="Alpha", source="seed://test", text="apple orchard evidence"),
        Document(doc_id="doc-002", title="Beta", source="seed://test", text="bond yield inflation"),
        Document(doc_id="doc-003", title="Gamma", source="seed://test", text="apple tree bloom"),
        Document(doc_id="doc-004", title="Delta", source="seed://test", text="market credit spread"),
    ]

    def fake_embed_texts(*, texts, **kwargs):
        del kwargs
        mapping = {
            "apple orchard evidence": [1.0, 0.0, 0.0, 0.0],
            "bond yield inflation": [0.0, 1.0, 0.0, 0.0],
            "apple tree bloom": [0.9, 0.1, 0.0, 0.0],
            "market credit spread": [0.0, 0.9, 0.1, 0.0],
            "apple question": [1.0, 0.0, 0.0, 0.0],
        }
        return [mapping[text] for text in texts], 0, 0.0

    monkeypatch.setattr("mirage.ingest.load_documents_for_spec", lambda runtime_spec: documents)
    monkeypatch.setattr("mirage.ingest.embed_texts", fake_embed_texts)
    monkeypatch.setattr("mirage.pipeline.embed_texts", fake_embed_texts)

    metadata = ingest_spec(spec, reset=True)
    retrieval, items = retrieve(spec, "apple question")

    assert metadata["store_backend_id"] == "faiss-local"
    assert metadata["store_index_variant_id"] == "idx-faiss-ivfflat-v1"
    assert items[0].doc_id in {"doc-001", "doc-003"}
    assert retrieval.retrieved_doc_ids[0] in {"doc-001", "doc-003"}


def test_faiss_ivfpq_ingest_and_retrieve(monkeypatch, tmp_path) -> None:
    spec = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "generation_model_id": "gen-llama-3.1-8b",
            "store_backend_id": "faiss-local",
            "store_index_variant_id": "idx-faiss-ivfpq-v1",
        },
    )[0].model_copy(
        update={"artifacts_dir": str(tmp_path / "artifacts"), "store_index_nlist": 2, "store_index_m": 2, "store_index_bits": 4}
    )

    documents = [
        Document(doc_id=f"doc-{idx:03d}", title=f"Doc {idx}", source="seed://test", text=f"topic {idx}")
        for idx in range(32)
    ]

    def fake_embed_texts(*, texts, **kwargs):
        del kwargs
        vectors = []
        for text in texts:
            if text == "finance question":
                vectors.append([0.0, 1.0, 0.0, 0.0])
                continue
            idx = int(text.split()[-1])
            if idx < 16:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0, 0.0])
        return vectors, 0, 0.0

    monkeypatch.setattr("mirage.ingest.load_documents_for_spec", lambda runtime_spec: documents)
    monkeypatch.setattr("mirage.ingest.embed_texts", fake_embed_texts)
    monkeypatch.setattr("mirage.pipeline.embed_texts", fake_embed_texts)

    metadata = ingest_spec(spec, reset=True)
    retrieval, items = retrieve(spec, "finance question")

    assert metadata["store_backend_id"] == "faiss-local"
    assert metadata["store_index_variant_id"] == "idx-faiss-ivfpq-v1"
    assert len(items) == spec.top_k
    assert retrieval.retrieved_doc_ids
