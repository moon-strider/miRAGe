---
report_id: rpt-01-rag-foundation-wave1-dense-topk10-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: wave1-dense-topk10
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: wave1-dense-topk10
---

## Goal

Compare dense top-k 10 against the dense top-k 5 frozen baseline on full SciFact retrieval metrics without LLM generation.

## Changed axis

- search algorithm: `search-dense-topk5-v1` -> `search-dense-topk10-v1`

## Metrics

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.8667 | +0.0667 |
| precision_at_k | 0.1727 | 0.0976 | -0.0751 |
| recall_at_k | 0.7771 | 0.8570 | +0.0799 |
| mrr_at_k | 0.6736 | 0.6826 | +0.0090 |
| ndcg_at_k | 0.6912 | 0.7193 | +0.0281 |
| p50 latency ms | 917.85 | 833.84 | -84.01 |
| p95 latency ms | 2067.89 | 1620.83 | -447.06 |

## Interpretation

Dense top-k 10 improves full SciFact retrieval over top-k 5: Hit@k rises by +0.0667, Recall@k by +0.0799, and NDCG@k by +0.0281. Precision@k drops because the candidate returns more documents per query.
