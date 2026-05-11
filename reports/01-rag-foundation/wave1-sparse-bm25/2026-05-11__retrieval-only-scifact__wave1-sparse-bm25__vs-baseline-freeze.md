---
report_id: rpt-01-rag-foundation-wave1-sparse-bm25-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: wave1-sparse-bm25
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: wave1-sparse-bm25
---

## Goal

Compare sparse BM25 retrieval against the dense top-k 5 frozen baseline on full SciFact without LLM calls.

## Metrics

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.6867 | -0.1133 |
| precision_at_k | 0.1727 | 0.0742 | -0.0985 |
| recall_at_k | 0.7771 | 0.6688 | -0.1083 |
| mrr_at_k | 0.6736 | 0.5095 | -0.1641 |
| ndcg_at_k | 0.6912 | 0.5438 | -0.1474 |
| p50 latency ms | 917.85 | 1722.37 | +804.52 |
| p95 latency ms | 2067.89 | 2005.89 | -62.00 |

## Interpretation

Sparse BM25 underperforms the dense baseline on full SciFact in this runtime. It is not competitive alone, but it remains useful as a component to interpret hybrid RRF.
