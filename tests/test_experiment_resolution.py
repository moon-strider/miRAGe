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
    assert {spec.dataset_adapter_id for spec in specs} == {"jsonl"}
    assert {spec.eval_adapter_id for spec in specs} == {"jsonl"}
    assert {spec.store_embedding_model_id for spec in specs} == {"emb-text-embedding-3-small"}
    assert {spec.query_embedding_model_id for spec in specs} == {"emb-text-embedding-3-small"}


def test_external_dataset_registry_resolves_adapter_roots() -> None:
    specs = load_experiment_specs(
        "experiments/01-rag-foundation/baseline-freeze",
        overrides={
            "dataset_id": "ds-beir-scifact-v1",
            "evalset_id": "ev-beir-scifact-v1",
            "generation_model_id": "gen-llama-3.1-8b",
        },
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.dataset_adapter_id == "scifact"
    assert spec.eval_adapter_id == "scifact"
    assert spec.dataset_root == "data/datasets/scifact"
    assert spec.eval_root == "data/datasets/scifact"
    assert spec.eval_split == "test"
    assert spec.corpus_path is None
    assert spec.eval_path is None


def test_deterministic_chunking_experiment_resolves_three_chunking_variants() -> None:
    specs = load_experiment_specs(
        "experiments/01-rag-foundation/load-deterministic-chunking",
        overrides={"generation_model_id": "gen-llama-3.1-8b"},
    )

    assert len(specs) == 3
    assert {spec.chunking_variant_id for spec in specs} == {
        "chunk-token-1024-128-v1",
        "chunk-token-768-128-v1",
        "chunk-sentence-1024-128-v1",
    }
    assert {spec.chunking_kind for spec in specs} == {"token", "sentence"}


def test_semantic_chunking_experiment_resolves_runtime_config() -> None:
    specs = load_experiment_specs(
        "experiments/01-rag-foundation/load-semantic-chunking",
        overrides={"generation_model_id": "gen-llama-3.1-8b"},
    )

    assert len(specs) == 3
    assert {spec.chunking_kind for spec in specs} == {"semantic"}
    assert {spec.chunking_model_id for spec in specs} == {
        "emb-bge-small-en-v1.5",
        "emb-text-embedding-3-small",
        "emb-text-embedding-3-large",
    }
    assert {spec.semantic_embedding_provider for spec in specs} == {"fastembed", "openrouter"}
    assert all(spec.chunk_size == 1024 for spec in specs)


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


def test_search_experiment_includes_dense_sparse_and_hybrid_variants() -> None:
    specs = load_experiment_specs(
        "experiments/01-rag-foundation/inference-search",
        overrides={"generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.search_kind for spec in specs} == {"dense", "sparse", "hybrid"}
    assert {spec.search_algorithm_id for spec in specs} == {
        "search-dense-topk5-v1",
        "search-dense-topk10-v1",
        "search-dense-topk20-v1",
        "search-sparse-bm25-topk10-v1",
        "search-hybrid-rrf-topk10-v1",
    }


def test_agentic_experiment_includes_reranker_axis() -> None:
    specs = load_experiment_specs(
        "experiments/01-rag-foundation/inference-agentic",
        overrides={"generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.reranker_id for spec in specs} == {"none", "rerank-minilm-l6-v1"}
    assert {spec.reranker_kind for spec in specs} == {"none", "cross-encoder"}


def test_artifact_layout_uses_layered_paths() -> None:
    spec = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")[0]
    layout = ArtifactLayout(spec)

    assert str(layout.prepared_dir()).endswith("runs/prepared/ds-docs-v1/prep-basic-clean-v1")
    assert str(layout.chunks_dir()).endswith(
        "runs/chunks/ds-docs-v1/prep-basic-clean-v1/chunk-token-1024-128-v1/none"
    )
    assert str(layout.store_dir()).endswith(
        "runs/store/ds-docs-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/"
        "store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small"
    )
