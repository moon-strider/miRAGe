from mirage.artifacts import ArtifactLayout
from mirage.config import load_experiment_specs


def test_baseline_experiment_resolves_three_generation_specs() -> None:
    specs = load_experiment_specs("studies/rag-foundation")

    assert len(specs) == 3
    assert {spec.generation_model_id for spec in specs} == {
        "gen-llama-3.1-8b",
        "gen-grok-4-fast",
        "gen-minimax-2.7",
    }
    assert {spec.chunking_model_id for spec in specs} == {"none"}
    assert {spec.dataset_adapter_id for spec in specs} == {"scifact"}
    assert {spec.eval_adapter_id for spec in specs} == {"scifact"}
    assert {spec.store_embedding_model_id for spec in specs} == {"emb-text-embedding-3-small"}
    assert {spec.query_embedding_model_id for spec in specs} == {"emb-text-embedding-3-small"}
    assert {spec.artifacts_dir for spec in specs} == {"artifacts"}


def test_study_wave1_retrieval_resolves_completed_experiments() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "dense-topk10", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert len(specs) == 1
    assert specs[0].group_id == "rag-foundation"
    assert specs[0].experiment_id == "dense-topk10"
    assert specs[0].search_algorithm_id == "search-dense-topk10-v1"


def test_wave1_experiment_directories_resolve_expected_axes() -> None:
    topk10 = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "dense-topk10", "generation_model_id": "gen-llama-3.1-8b"},
    )[0]
    hybrid = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "hybrid-rrf", "generation_model_id": "gen-llama-3.1-8b"},
    )[0]
    rerank = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "rerank-jina-tiny", "generation_model_id": "gen-llama-3.1-8b"},
    )[0]
    semantic = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "semantic-chunking", "generation_model_id": "gen-llama-3.1-8b"},
    )[0]
    embedding_boundary = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "study_experiment_id": "embedding-boundary-chunking",
            "generation_model_id": "gen-llama-3.1-8b",
            "chunking_model_id": "emb-text-embedding-3-small",
            "semantic_similarity_threshold": 0.85,
            "semantic_min_sentences_per_chunk": 2,
        },
    )[0]
    tools = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "tool-context-expansion", "generation_model_id": "gen-llama-3.1-8b"},
    )[0]

    assert topk10.search_algorithm_id == "search-dense-topk10-v1"
    assert hybrid.search_algorithm_id == "search-hybrid-rrf-topk10-v1"
    assert rerank.search_algorithm_id == "search-dense-topk10-v1"
    assert rerank.reranker_id == "rerank-jina-tiny-v1"
    assert semantic.chunking_kind == "semantic"
    assert semantic.chunking_model_id == "gen-nemotron-3-nano-omni-free"
    assert semantic.semantic_chunking_provider == "openrouter"
    assert embedding_boundary.chunking_kind == "embedding-boundary"
    assert embedding_boundary.chunking_model_id == "emb-text-embedding-3-small"
    assert embedding_boundary.semantic_similarity_threshold == 0.85
    assert embedding_boundary.semantic_min_sentences_per_chunk == 2
    assert tools.tool_policy_id == "tool-context-expansion-v1"


def test_external_dataset_registry_resolves_adapter_roots() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
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
    assert spec.dataset_root == "datasets/scifact"
    assert spec.eval_root == "datasets/scifact"
    assert spec.eval_split == "test"
    assert spec.corpus_path is None
    assert spec.eval_path is None


def test_deterministic_chunking_experiment_resolves_llm_free_macro_variants() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "load-deterministic-chunking", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert len(specs) == 4
    assert {spec.chunking_variant_id for spec in specs} == {
        "chunk-token-1024-128-v1",
        "chunk-token-512-64-v1",
        "chunk-token-2048-256-v1",
        "chunk-sentence-1024-128-v1",
    }
    assert {spec.chunking_kind for spec in specs} == {"token", "sentence"}
    assert {spec.store_embedding_model_id for spec in specs} == {"emb-gemini-embedding-001"}
    assert {spec.query_embedding_model_id for spec in specs} == {"emb-gemini-embedding-001"}
    assert {spec.tool_policy_id for spec in specs} == {"none"}


def test_semantic_chunking_experiment_resolves_single_llm_planning_preset() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "semantic-chunking", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.chunking_kind == "semantic"
    assert spec.chunking_model_id == "gen-nemotron-3-nano-omni-free"
    assert spec.semantic_chunking_provider == "openrouter"
    assert spec.semantic_chunking_model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    assert spec.semantic_embedding_provider == "none"
    assert "prompt-llm-semantic-boundary-v1" in spec.load_variant_id
    assert "schema-llm-chunk-plan-v1" in spec.load_variant_id
    assert "max-1024" in spec.load_variant_id
    assert spec.chunk_size == 1024
    assert spec.tool_policy_id == "none"


def test_embedding_boundary_threshold_changes_load_artifact_key() -> None:
    low = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "study_experiment_id": "embedding-boundary-chunking",
            "generation_model_id": "none",
            "semantic_similarity_threshold": 0.65,
        },
    )[0]
    high = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "study_experiment_id": "embedding-boundary-chunking",
            "generation_model_id": "none",
            "semantic_similarity_threshold": 0.75,
        },
    )[0]

    assert low.load_variant_id != high.load_variant_id
    assert "th-0p65" in low.load_variant_id
    assert "th-0p75" in high.load_variant_id


def test_controlled_embedding_boundary_experiment_resolves_three_cases() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "embedding-boundary-controlled", "generation_model_id": "none"},
    )

    assert len(specs) == 3
    assert {spec.chunking_kind for spec in specs} == {"embedding-boundary"}
    assert {spec.search_algorithm_id for spec in specs} == {"search-dense-topk10-v1"}
    assert {spec.store_embedding_model_id for spec in specs} == {"emb-gemini-embedding-001"}
    assert {
        (spec.chunking_model_id, spec.semantic_similarity_threshold)
        for spec in specs
    } == {
        ("emb-gemini-embedding-001", 0.75),
        ("emb-gemini-embedding-001", 0.65),
        ("emb-text-embedding-3-small", 0.75),
    }


def test_embedding_experiment_uses_coupled_cases() -> None:
    specs = load_experiment_specs("studies/rag-foundation", overrides={"study_experiment_id": "store-embedding-models"})

    pairs = {
        (spec.store_embedding_model_id, spec.query_embedding_model_id)
        for spec in specs
    }
    assert pairs == {
        ("emb-text-embedding-3-small", "emb-text-embedding-3-small"),
        ("emb-text-embedding-3-large", "emb-text-embedding-3-large"),
        ("emb-text-embedding-ada-002", "emb-text-embedding-ada-002"),
        ("emb-pplx-embed-v1-0.6b", "emb-pplx-embed-v1-0.6b"),
        ("emb-pplx-embed-v1-4b", "emb-pplx-embed-v1-4b"),
        ("emb-gemini-embedding-001", "emb-gemini-embedding-001"),
    }


def test_search_experiment_includes_dense_sparse_and_hybrid_variants() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "inference-search", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.search_kind for spec in specs} == {"dense", "sparse", "hybrid", "dense-mmr"}
    assert {spec.search_algorithm_id for spec in specs} == {
        "search-dense-topk5-v1",
        "search-dense-topk10-v1",
        "search-dense-topk20-v1",
        "search-sparse-bm25-topk10-v1",
        "search-hybrid-rrf-topk10-v1",
        "search-dense-mmr-topk10-v1",
    }


def test_prompting_experiment_includes_four_grounded_prompt_variants() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "inference-prompting", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.prompt_variant_id for spec in specs} == {
        "prompt-grounded-citations-v1",
        "prompt-brief-grounded-v1",
        "prompt-evidence-first-v1",
        "prompt-strict-abstain-v1",
    }


def test_free_openrouter_generation_models_resolve_zero_pricing() -> None:
    ids = {
        "gen-nemotron-3-nano-omni-free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "gen-gemma-4-31b-free": "google/gemma-4-31b-it:free",
        "gen-nemotron-3-super-free": "nvidia/nemotron-3-super-120b-a12b:free",
        "gen-minimax-m2.5-free": "minimax/minimax-m2.5:free",
        "gen-gpt-oss-120b-free": "openai/gpt-oss-120b:free",
    }

    for model_id, provider_model in ids.items():
        spec = load_experiment_specs("studies/rag-foundation", overrides={"generation_model_id": model_id})[0]
        assert spec.generation_model == provider_model
        assert spec.generation_pricing_input_per_1m_tokens_usd == 0.0
        assert spec.generation_pricing_output_per_1m_tokens_usd == 0.0


def test_load_preprocessing_experiment_exposes_three_runtime_variants() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "load-preprocessing", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.preprocessing_variant_id for spec in specs} == {
        "prep-basic-clean-v1",
        "prep-basic-clean-dedupe-v1",
        "prep-basic-clean-metadata-v1",
    }


def test_agentic_experiment_includes_reranker_axis() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "inference-agentic", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.reranker_id for spec in specs} == {
        "none",
        "rerank-minilm-l6-v1",
        "rerank-minilm-l12-v1",
        "rerank-jina-tiny-v1",
        "rerank-jina-turbo-v1",
    }
    assert {spec.reranker_kind for spec in specs} == {"none", "cross-encoder"}


def test_controlled_reranking_experiment_uses_candidate_pool() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "reranking-controlled", "generation_model_id": "none"},
    )

    assert len(specs) == 4
    assert {spec.search_algorithm_id for spec in specs} == {"search-dense-candidates50-topk10-v1"}
    assert {spec.top_k for spec in specs} == {10}
    assert {spec.search_dense_top_k for spec in specs} == {50}
    assert {spec.reranker_id for spec in specs} == {
        "rerank-bge-base-v1",
        "rerank-jina-v2-base-multilingual",
        "rerank-cohere-4-fast-openrouter",
        "rerank-cohere-4-pro-openrouter",
    }
    assert {spec.reranker_kind for spec in specs} == {"cross-encoder", "openrouter-rerank"}


def test_store_backend_experiment_includes_faiss_runtime() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "store-backends", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.store_backend_id for spec in specs} == {"qdrant", "faiss-local"}
    assert {spec.store_backend_runtime_status for spec in specs} == {"active"}


def test_faiss_index_variants_expose_runtime_knobs() -> None:
    ivfflat = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "study_experiment_id": "store-backends",
            "generation_model_id": "gen-llama-3.1-8b",
            "store_backend_id": "faiss-local",
            "store_index_variant_id": "idx-faiss-ivfflat-v1",
        },
    )[0]
    ivfpq = load_experiment_specs(
        "studies/rag-foundation",
        overrides={
            "study_experiment_id": "store-backends",
            "generation_model_id": "gen-llama-3.1-8b",
            "store_backend_id": "faiss-local",
            "store_index_variant_id": "idx-faiss-ivfpq-v1",
        },
    )[0]

    assert ivfflat.store_index_runtime_status == "active"
    assert ivfflat.store_index_nlist == 256
    assert ivfpq.store_index_runtime_status == "active"
    assert ivfpq.store_index_nlist == 256
    assert ivfpq.store_index_m == 16
    assert ivfpq.store_index_bits == 8


def test_store_index_experiment_keeps_active_qdrant_variants() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "store-index-variants", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.store_index_runtime_status for spec in specs} == {"active"}


def test_agentic_experiment_tool_policies_are_active_runtime_axes() -> None:
    specs = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "inference-agentic", "generation_model_id": "gen-llama-3.1-8b"},
    )

    assert {spec.tool_policy_id for spec in specs} == {
        "none",
        "tool-react-v1",
        "tool-context-expansion-v1",
    }


def test_artifact_layout_uses_layered_paths() -> None:
    spec = load_experiment_specs("studies/rag-foundation")[0]
    layout = ArtifactLayout(spec)

    assert str(layout.prepared_dir()).endswith("artifacts/prepared/ds-beir-scifact-v1/prep-basic-clean-v1")
    assert str(layout.chunks_dir()).endswith(
        "artifacts/chunks/ds-beir-scifact-v1/prep-basic-clean-v1/chunk-token-1024-128-v1/none"
    )
    assert str(layout.store_dir()).endswith(
        "artifacts/store/ds-beir-scifact-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/"
        "store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small"
    )
