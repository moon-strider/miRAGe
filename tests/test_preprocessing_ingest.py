from __future__ import annotations

from pathlib import Path

from mirage.config import load_experiment_specs
from mirage.ingest import prepare_documents
from mirage.schemas import Document


def test_prepare_documents_applies_dedupe_preprocessing(monkeypatch, tmp_path: Path) -> None:
    source_documents = [
        Document(doc_id="doc-001", title="Alpha", source="seed://test", text="Same text"),
        Document(doc_id="doc-002", title="Alpha", source="seed://test", text="Same text"),
        Document(doc_id="doc-003", title="Beta", source="seed://test", text="Different text"),
    ]

    spec = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "generation_model_id": "gen-llama-3.1-8b",
            "preprocessing_variant_id": "prep-basic-clean-dedupe-v1",
        },
    )[0].model_copy(update={"artifacts_dir": str(tmp_path / "artifacts")})

    monkeypatch.setattr("mirage.ingest.load_documents_for_spec", lambda runtime_spec: source_documents)

    documents, info = prepare_documents(spec, reset=True)

    assert info["documents"] == 2
    assert [document.doc_id for document in documents] == ["doc-001", "doc-003"]


def test_prepare_documents_applies_metadata_preprocessing(monkeypatch, tmp_path: Path) -> None:
    source_documents = [
        Document(
            doc_id="doc-001",
            title="Alpha",
            source="seed://test",
            text="Body text",
            metadata={"section": "intro"},
        )
    ]

    spec = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "generation_model_id": "gen-llama-3.1-8b",
            "preprocessing_variant_id": "prep-basic-clean-metadata-v1",
        },
    )[0].model_copy(update={"artifacts_dir": str(tmp_path / "artifacts")})

    monkeypatch.setattr("mirage.ingest.load_documents_for_spec", lambda runtime_spec: source_documents)

    documents, info = prepare_documents(spec, reset=True)

    assert info["documents"] == 1
    assert documents[0].text.startswith("Title: Alpha")
    assert "Metadata: {\"section\": \"intro\"}" in documents[0].text
