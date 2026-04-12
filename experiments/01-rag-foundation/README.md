# 01-rag-foundation

This group defines the first large family of RAG experiments in the repository.

It is organized by reusable variable families instead of creating a separate
pipeline for every matrix point.

## Scope

The group covers four variation families:

- load-time variants: preprocessing and chunking
- store-time variants: embedding models, index variants, and backends
- inference-time variants: search, prompting, and tool policies
- generation-time variants: the fixed comparison set of three generation models

## Fixed comparison set

Unless an experiment overlay overrides it, this group uses:

- `dataset_id = ds-docs-v1`
- `evalset_id = ev-dev-v1`
- `store_backend_id = qdrant`
- `store_index_variant_id = idx-qdrant-hnsw-cosine-default-v1`
- `search_algorithm_id = search-dense-topk5-v1`
- `reranker_id = none`
- `prompt_variant_id = prompt-grounded-citations-v1`
- `tool_policy_id = none`
- `generation_model_id ∈ {gen-llama-3.1-8b, gen-grok-4-fast, gen-minimax-2.7}`

## Experiment directories

- `baseline-freeze/`
- `load-preprocessing/`
- `load-deterministic-chunking/`
- `load-semantic-chunking/`
- `store-embedding-models/`
- `store-index-variants/`
- `store-backends/`
- `inference-search/`
- `inference-prompting/`
- `inference-agentic/`

Some experiment directories may describe variants that are already defined in registries but not yet executable in runtime. Their `README.md` files must state that status explicitly.
