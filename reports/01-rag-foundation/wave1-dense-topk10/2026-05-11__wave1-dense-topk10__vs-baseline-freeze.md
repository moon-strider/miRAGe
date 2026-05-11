---
report_id: rpt-01-rag-foundation-wave1-dense-topk10-wave1-dense-topk10-vs-baseline-freeze
created_at: 2026-05-11T13:48:02Z
group_id: 01-rag-foundation
experiment_id: wave1-dense-topk10
baseline_id: baseline-freeze
candidate_id: wave1-dense-topk10
dataset_id: ds-docs-v1
evalset_id: ev-dev-v1
preprocessing_variant_id: prep-basic-clean-v1
chunking_variant_id: chunk-token-1024-128-v1
chunking_model_id: none
load_variant_id: load-prep-basic-clean-v1__chunk-token-1024-128-v1__none
store_backend_id: qdrant
store_index_variant_id: idx-qdrant-hnsw-cosine-default-v1
store_embedding_model_id: emb-text-embedding-3-small
query_embedding_model_id: emb-text-embedding-3-small
search_algorithm_id: search-dense-topk10-v1
reranker_id: none
prompt_variant_id: prompt-grounded-citations-v1
tool_policy_id: none
inference_variant_id: inf-emb-text-embedding-3-small__search-dense-topk10-v1__none__prompt-grounded-citations-v1__none
generation_model_ids:
  - gen-grok-4-fast
  - gen-llama-3.1-8b
  - gen-minimax-2.7
run_paths:
  - /home/master/Documents/Projects/miRAGe/runs/answers/ds-docs-v1/ev-dev-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small/inf-emb-text-embedding-3-small__search-dense-topk10-v1__none__prompt-grounded-citations-v1__none/gen-grok-4-fast
  - /home/master/Documents/Projects/miRAGe/runs/answers/ds-docs-v1/ev-dev-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small/inf-emb-text-embedding-3-small__search-dense-topk10-v1__none__prompt-grounded-citations-v1__none/gen-llama-3.1-8b
  - /home/master/Documents/Projects/miRAGe/runs/answers/ds-docs-v1/ev-dev-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small/inf-emb-text-embedding-3-small__search-dense-topk10-v1__none__prompt-grounded-citations-v1__none/gen-minimax-2.7
price_catalog_version: openrouter-public-v1
---

## Goal

State the measured difference between the named baseline and candidate using the saved experiment artifacts.

## Compared variants

| Axis | Baseline | Candidate | Status |
| --- | --- | --- | --- |
| preprocessing | prep-basic-clean-v1 | prep-basic-clean-v1 | fixed |
| chunking | chunk-token-1024-128-v1 | chunk-token-1024-128-v1 | fixed |
| chunking model | none | none | fixed |
| store backend | qdrant | qdrant | fixed |
| index variant | idx-qdrant-hnsw-cosine-default-v1 | idx-qdrant-hnsw-cosine-default-v1 | fixed |
| store embedding | emb-text-embedding-3-small | emb-text-embedding-3-small | fixed |
| query embedding | emb-text-embedding-3-small | emb-text-embedding-3-small | fixed |
| search algorithm | search-dense-topk5-v1 | search-dense-topk10-v1 | changed |
| reranker | none | none | fixed |
| prompt variant | prompt-grounded-citations-v1 | prompt-grounded-citations-v1 | fixed |
| tool policy | none | none | fixed |
| generation models | see metrics table | see metrics table | changed |

## Artifact reuse

| Layer | Artifact id or path | Reused or rebuilt | Why |
| --- | --- | --- | --- |
| prepared | /home/master/Documents/Projects/miRAGe/runs/prepared/ds-docs-v1/prep-basic-clean-v1 | rebuilt | baseline freeze materialized this layer |
| chunks | /home/master/Documents/Projects/miRAGe/runs/chunks/ds-docs-v1/prep-basic-clean-v1/chunk-token-1024-128-v1/none | rebuilt | baseline freeze materialized this layer |
| store | /home/master/Documents/Projects/miRAGe/runs/store/ds-docs-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small | rebuilt | baseline freeze materialized this layer |
| retrieval | /home/master/Documents/Projects/miRAGe/runs/retrieval/ds-docs-v1/ev-dev-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small/emb-text-embedding-3-small/search-dense-topk5-v1/none | rebuilt | baseline freeze materialized this layer |
| answers | /home/master/Documents/Projects/miRAGe/runs/answers/ds-docs-v1/ev-dev-v1/load-prep-basic-clean-v1__chunk-token-1024-128-v1__none/store-qdrant__idx-qdrant-hnsw-cosine-default-v1__emb-text-embedding-3-small/inf-emb-text-embedding-3-small__search-dense-topk5-v1__none__prompt-grounded-citations-v1__none/gen-grok-4-fast | rebuilt | baseline freeze materialized this layer |

## Metrics summary

| generation_model_id | baseline Hit@k | candidate Hit@k | baseline Precision@k | candidate Precision@k | baseline Recall@k | candidate Recall@k | baseline MRR@k | candidate MRR@k | baseline NDCG@k | candidate NDCG@k | baseline Exact Match | candidate Exact Match | baseline Token F1 | candidate Token F1 | baseline Citation Hit Rate | candidate Citation Hit Rate | baseline p50 ms | candidate p50 ms | baseline p95 ms | candidate p95 ms | baseline reranker coverage | candidate reranker coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gen-grok-4-fast | 1.0000 | 1.0000 | 0.2000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4405 | 0.4405 | 1.0000 | 1.0000 | 3788.07 | 3750.25 | 4793.09 | 19036.12 | 1.0000 | 1.0000 |
| gen-llama-3.1-8b | 1.0000 | 1.0000 | 0.2000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4500 | 0.4500 | 1.0000 | 1.0000 | 2010.30 | 1915.35 | 2551.16 | 2042.72 | 1.0000 | 1.0000 |
| gen-minimax-2.7 | 1.0000 | 1.0000 | 0.2000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1984 | 0.1984 | 1.0000 | 1.0000 | 3276.63 | 4197.28 | 5516.66 | 5647.75 | 1.0000 | 1.0000 |

## Cost summary

| generation_model_id | baseline build preprocessing | candidate build preprocessing | baseline build chunking | candidate build chunking | baseline build store embeddings | candidate build store embeddings | baseline mean query cost over eval | candidate mean query cost over eval | baseline projected query cost at 1M | candidate projected query cost at 1M | baseline amortized total at 1M | candidate amortized total at 1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gen-grok-4-fast | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000005 | 0.000005 | 0.000196 | 0.000195 | 195.60 | 194.60 | 195.60 | 194.60 |
| gen-llama-3.1-8b | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000005 | 0.000005 | 0.000010 | 0.000010 | 9.60 | 9.60 | 9.60 | 9.60 |
| gen-minimax-2.7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000005 | 0.000005 | 0.000264 | 0.000264 | 263.60 | 263.60 | 263.60 | 263.60 |

## Failure cases

- qid: q-001
  question: What is the main command entrypoint for common workflows?
  expected answer summary: just
  retrieved document ids: doc-002
  model answer summary: just [doc-002]
  why the case matters: synthesized representative case from saved artifacts
- qid: q-002
  question: Which tool manages Python dependencies in the baseline repository conventions?
  expected answer summary: uv
  retrieved document ids: doc-002
  model answer summary: **uv** manages Python dependencies in the baseline repository conventions [doc-002].
  why the case matters: synthesized representative case from saved artifacts

## Conclusion

wave1-dense-topk10 matched the baseline on mean exact-match performance in the saved runs, so the decision depends on latency and cost tradeoffs.
