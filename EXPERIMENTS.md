# Experiments Contract

## Purpose

This document is the canonical contract for experiment structure in this repository.
It defines the allowed axes, how configuration is composed, how artifacts are named,
how reuse is determined, and how reports link back to machine-readable outputs.

Concrete studies do not live in this document.
Concrete studies live under `experiments/`.

## Canonical atomic axes

Every evaluated point in the matrix is defined by these atomic axes:

- `dataset_id`
- `evalset_id`
- `preprocessing_variant_id`
- `chunking_variant_id`
- `chunking_model_id`
- `store_backend_id`
- `store_index_variant_id`
- `store_embedding_model_id`
- `query_embedding_model_id`
- `search_algorithm_id`
- `reranker_id`
- `prompt_variant_id`
- `tool_policy_id`
- `generation_model_id`

`chunking_model_id` is mandatory.
Use the literal value `none` when chunking is not model-driven.

## Composite identifiers

The system derives these composite identifiers from the atomic axes:

- `load_variant_id = load-<preprocessing_variant_id>__<chunking_variant_id>__<chunking_model_id>`
- `store_variant_id = store-<store_backend_id>__<store_index_variant_id>__<store_embedding_model_id>`
- `inference_variant_id = inf-<query_embedding_model_id>__<search_algorithm_id>__<reranker_id>__<prompt_variant_id>__<tool_policy_id>`

A `run` is one concrete point in the full matrix:

`dataset × evalset × load axes × store axes × inference axes × generation model`

## Experiment directory contract

Concrete experiments live under:

```text
experiments/<group_id>/<experiment_id>/
```

Each experiment directory must contain:

- `README.md` — human-readable experiment definition
- `overlay.toml` — machine-readable experiment overlay

Each experiment directory may contain:

- `hooks.py` — optional experiment-local code for extension points that cannot be expressed through registries and overlays alone

Each experiment group may contain:

- `README.md` — group-level description
- `group.toml` — group-level shared controls and matrix definitions

## Configuration composition order

Runtime configuration is composed in this exact order:

1. `configs/system.toml`
2. all registry files under `configs/registries/`
3. selected prototype files under `configs/prototypes/`
4. `experiments/<group_id>/group.toml`
5. `experiments/<group_id>/<experiment_id>/overlay.toml`
6. CLI key-value overrides
7. optional `hooks.py`

Later layers override earlier layers.

## Registry families

The repository keeps experiment variables in registries.
Each registry family is addressed by stable ids.

Required registry families:

- datasets
- evalsets
- preprocessing variants
- chunking variants
- embedding models
- generation models
- store backends
- store index variants
- search algorithms
- prompt variants
- rerankers
- tool policies

## Prototype families

Prototypes are reusable bundles of fixed selections.
They reduce duplication across experiments without hiding variables.

Required prototype families:

- `load`
- `store`
- `inference`

A prototype may set any fixed fields that belong to its phase.
A prototype does not replace explicit atomic axes in the resolved spec.

## Overlay contract

Each `overlay.toml` may contain these top-level tables:

- `[experiment]`
- `[extends]`
- `[fixed]`
- `[matrix]`
- `[[cases]]`

### `[experiment]`

Required fields:

- `group_id`
- `experiment_id`

### `[extends]`

Allowed keys:

- `load`
- `store`
- `inference`

Values are prototype ids.

### `[fixed]`

`[fixed]` contains axis assignments or runtime knobs that apply to every run
expanded from the experiment.

### `[matrix]`

`[matrix]` contains explicit lists of values.
The runtime expands the cartesian product of these lists into concrete specs.

### `[[cases]]`

`[[cases]]` is an optional list of coupled assignments.
Each case is merged on top of `[fixed]` and on top of each expanded matrix point.
Use it when several variables must move together and must not be expanded independently.

## Artifact layers

The system stores machine-readable artifacts in these layers:

1. `prepared`
2. `chunks`
3. `store`
4. `retrieval`
5. `answers`

The canonical path structure is:

```text
runs/
  prepared/<dataset_id>/<preprocessing_variant_id>/
  chunks/<dataset_id>/<preprocessing_variant_id>/<chunking_variant_id>/<chunking_model_id>/
  store/<dataset_id>/<load_variant_id>/<store_variant_id>/
  retrieval/<dataset_id>/<evalset_id>/<load_variant_id>/<store_variant_id>/<query_embedding_model_id>/<search_algorithm_id>/<reranker_id>/
  answers/<dataset_id>/<evalset_id>/<load_variant_id>/<store_variant_id>/<inference_variant_id>/<generation_model_id>/
```

Each terminal artifact directory must contain `resolved_spec.json`.

## Reuse and invalidation rules

| Change | Reuse allowed | Rebuild required |
| --- | --- | --- |
| dataset contents | nothing downstream | all artifact layers |
| preprocessing only | source dataset | prepared onward |
| chunking variant only | prepared | chunks onward |
| semantic chunking model only | prepared | chunks onward |
| store embedding model only | chunks | store onward |
| store backend only | chunks and embeddings when backend accepts the same vectors | store onward |
| store index parameters only | chunks and embeddings | store onward |
| query embedding only | prepared, chunks, store | retrieval onward |
| search algorithm only | prepared, chunks, store | retrieval onward |
| reranker only | prepared, chunks, store | retrieval onward |
| prompt only | prepared, chunks, store, retrieval | answers onward |
| tool policy only | prepared, chunks, store | answers onward |
| generation model only | prepared, chunks, store, retrieval | answers onward |

Semantic chunking is model-bound.
Artifacts produced with one `chunking_model_id` are never interchangeable with artifacts produced with another `chunking_model_id`.

## Cost accounting contract

Every evaluation output and every report must keep external cost accounting split into two buckets.

### One-time build cost

The build bucket covers work paid once per dataset or per rebuild.
Required fields:

- `build_preprocessing_cost_usd`
- `build_chunking_cost_usd`
- `build_store_embedding_cost_usd`
- `build_total_external_cost_usd`

### Recurring query cost

The query bucket covers work paid per request.
Required fields:

- `query_embedding_cost_usd`
- `retrieval_rerank_cost_usd`
- `generation_input_cost_usd`
- `generation_output_cost_usd`
- `query_total_external_cost_usd`

### Required rollups

Required report and summary rollups:

- `mean_query_cost_usd_eval`
- `projected_query_cost_1m_usd`
- `one_time_build_cost_usd`
- `amortized_total_cost_1m_usd`

## Pinned price catalog

The repository uses this pinned external price catalog identifier:

- `price_catalog_version = openrouter-public-v1`

Pinned generation models:

| generation_model_id | OpenRouter slug | Input $ / 1M tokens | Output $ / 1M tokens |
| --- | --- | ---: | ---: |
| `gen-llama-3.1-8b` | `meta-llama/llama-3.1-8b-instruct` | 0.02 | 0.05 |
| `gen-grok-4-fast` | `x-ai/grok-4-fast` | 0.20 | 0.50 |
| `gen-minimax-2.7` | `minimax/minimax-m2.7` | 0.30 | 1.20 |

Pinned embedding models:

| embedding_model_id | Runtime | Model identifier | Input $ / 1M tokens |
| --- | --- | --- | ---: |
| `emb-bge-small-en-v1.5` | local | `BAAI/bge-small-en-v1.5` | 0.00 |
| `emb-text-embedding-3-small` | openrouter | `openai/text-embedding-3-small` | 0.02 |
| `emb-text-embedding-3-large` | openrouter | `openai/text-embedding-3-large` | 0.13 |

Pinned source pages:

- `https://openrouter.ai/meta-llama/llama-3.1-8b-instruct/pricing`
- `https://openrouter.ai/x-ai/grok-4-fast/api`
- `https://openrouter.ai/minimax/minimax-m2.7`
- `https://openrouter.ai/openai/text-embedding-3-small`
- `https://openrouter.ai/openai/text-embedding-3-large`

## Report linkage contract

Human-readable reports live under `reports/`.
Each report must follow `reports/SCHEMA.md` and link back to machine-readable artifacts in `runs/`.
Reports do not define system behavior.
Reports summarize measured behavior.

## Naming rules

Use lowercase kebab-case identifiers.
Versioned identifiers must carry an explicit suffix such as `-v1`.

Examples:

- `dataset_id = ds-docs-v1`
- `evalset_id = ev-dev-v1`
- `preprocessing_variant_id = prep-basic-clean-v1`
- `chunking_variant_id = chunk-token-500-50-v1`
- `chunking_model_id = none`
- `store_backend_id = qdrant`
- `store_index_variant_id = idx-qdrant-hnsw-cosine-default-v1`
- `store_embedding_model_id = emb-bge-small-en-v1.5`
- `query_embedding_model_id = emb-text-embedding-3-small`
- `search_algorithm_id = search-dense-topk5-v1`
- `prompt_variant_id = prompt-grounded-citations-v1`
- `tool_policy_id = none`
- `generation_model_id = gen-llama-3.1-8b`

Stable ids are part of the experimental record.
Compatibility-breaking changes require a new versioned id.
