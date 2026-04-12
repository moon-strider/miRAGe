from mirage.artifacts import ArtifactLayout
from mirage.config import load_experiment_specs


def test_baseline_experiment_resolves_three_generation_specs() -> None:
    specs = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")

    assert len(specs) == 3
    assert {spec.generation_model_id for spec in specs} == {
        "gen-llama-3.1-8b",
        "gen-grok-4-fast",
        "gen-minimax-2.7",
    }
    assert {spec.chunking_model_id for spec in specs} == {"none"}


def test_embedding_experiment_uses_coupled_cases() -> None:
    specs = load_experiment_specs("experiments/01-rag-foundation/store-embedding-models")

    pairs = {
        (spec.store_embedding_model_id, spec.query_embedding_model_id)
        for spec in specs
    }
    assert pairs == {
        ("emb-bge-small-en-v1.5", "emb-bge-small-en-v1.5"),
        ("emb-text-embedding-3-small", "emb-text-embedding-3-small"),
        ("emb-text-embedding-3-large", "emb-text-embedding-3-large"),
    }


def test_artifact_layout_uses_layered_paths() -> None:
    spec = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")[0]
    layout = ArtifactLayout(spec)

    assert str(layout.prepared_dir()).endswith("runs/prepared/ds-docs-v1/prep-basic-clean-v1")
    assert str(layout.chunks_dir()).endswith(
        "runs/chunks/ds-docs-v1/prep-basic-clean-v1/chunk-token-500-50-v1/none"
    )
    assert str(layout.store_dir()).endswith(
        "runs/store/ds-docs-v1/load-prep-basic-clean-v1__chunk-token-500-50-v1__none/"
        "store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-bge-small-en-v1.5"
    )
