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
- `store_embedding_model_id = emb-text-embedding-3-small`
- `query_embedding_model_id = emb-text-embedding-3-small`
- `search_algorithm_id = search-dense-topk5-v1`
- `reranker_id = none`
- `prompt_variant_id = prompt-grounded-citations-v1`
- `tool_policy_id = none`
- `generation_model_id ∈ {gen-llama-3.1-8b, gen-grok-4-fast, gen-minimax-2.7}`

The committed `ds-docs-v1` fixture is a smoke-test dataset only.
External benchmark sources for real runs are cataloged under `data/datasets/`.
Their adapters map upstream payloads into repository `Document` and `EvalExample`
records at runtime; no repository-local normalized benchmark copies are stored.

## Experiment directories

- `baseline-freeze/` — executable benchmark baseline with parameter-by-parameter rationale
- `load-preprocessing/`
- `load-deterministic-chunking/`
- `load-semantic-chunking/`
- `store-embedding-models/`
- `store-index-variants/`
- `store-backends/`
- `inference-search/`
- `inference-prompting/`
- `inference-agentic/`
- `wave1-dense-topk10/` — wider dense retrieval depth against the baseline
- `wave1-hybrid-rrf/` — hybrid sparse+dense retrieval against the baseline
- `wave1-dense-topk3/` — narrower dense retrieval depth against the baseline
- `wave1-dense-topk10/` — wider dense retrieval depth against the baseline
- `wave1-dense-topk20/` — high-recall dense retrieval depth against the baseline
- `wave1-sparse-bm25/` — sparse lexical retrieval against the baseline
- `wave1-hybrid-rrf/` — hybrid sparse+dense retrieval against the baseline
- `wave1-rerank-jina-tiny/` — planned later reranker/LLM-stage candidate
- `wave1-semantic-chunking/` — planned later embedding-heavy chunking candidate
- `wave1-tool-context-expansion/` — planned later orchestration candidate

Some experiment directories may describe variants that are already defined in registries but not yet executable in runtime. Their `README.md` files must state that status explicitly.


## Wave1 retrieval-only SciFact summary

These results use the full `ds-beir-scifact-v1` / `ev-beir-scifact-v1` benchmark path and make no LLM generation calls.
The baseline is `baseline-freeze` with `search-dense-topk5-v1`.

Baseline metrics:
- Hit@k: 0.8000
- Precision@k: 0.1727
- Recall@k: 0.7771
- MRR@k: 0.6742
- NDCG@k: 0.6917
- p50 latency: 922.37 ms

| experiment | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wave1-dense-topk3` | 0.7467 (-0.0533) | 0.2644 (+0.0917) | 0.7224 (-0.0547) | 0.6611 (-0.0131) | 0.6669 (-0.0248) | 911.00 (-11.37) |
| `wave1-dense-topk10` | 0.8667 (+0.0667) | 0.0976 (-0.0751) | 0.8570 (+0.0799) | 0.6826 (+0.0084) | 0.7193 (+0.0276) | 833.84 (-88.53) |
| `wave1-dense-topk20` | 0.9000 (+0.1000) | 0.0513 (-0.1214) | 0.8977 (+0.1206) | 0.6849 (+0.0107) | 0.7301 (+0.0384) | 919.28 (-3.09) |
| `wave1-sparse-bm25` | 0.6867 (-0.1133) | 0.0742 (-0.0985) | 0.6688 (-0.1083) | 0.5095 (-0.1647) | 0.5438 (-0.1479) | 1722.37 (+800.00) |
| `wave1-hybrid-rrf` | 0.8467 (+0.0467) | 0.0935 (-0.0792) | 0.8321 (+0.0550) | 0.6530 (-0.0212) | 0.6899 (-0.0018) | 1788.07 (+865.70) |

Interpretation:
- `wave1-dense-topk20` is the strongest recall-oriented candidate: it reaches Hit@k 0.9000 and Recall@k 0.8977.
- `wave1-dense-topk10` is a good middle point: it improves recall materially with less context expansion than top-k 20.
- `wave1-dense-topk3` confirms top-k 5 is not obviously over-retrieving: precision improves, but recall and hit rate drop.
- `wave1-sparse-bm25` underperforms dense retrieval alone on SciFact.
- `wave1-hybrid-rrf` improves over baseline but underperforms dense top-k 10/top-k 20 in this run, so hybrid should not become the default yet.
