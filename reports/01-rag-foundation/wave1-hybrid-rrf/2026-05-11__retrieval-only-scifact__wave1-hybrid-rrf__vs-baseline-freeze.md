---
report_id: rpt-01-rag-foundation-wave1-hybrid-rrf-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: wave1-hybrid-rrf
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: wave1-hybrid-rrf
---

## Goal

Compare hybrid BM25+dense RRF retrieval against the dense top-k 5 frozen baseline on full SciFact without LLM calls.

## Changed axis

- search algorithm: `search-dense-topk5-v1` -> `search-hybrid-rrf-topk10-v1`

## Metrics

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.8467 | +0.0467 |
| precision_at_k | 0.1727 | 0.0935 | -0.0792 |
| recall_at_k | 0.7771 | 0.8321 | +0.0550 |
| mrr_at_k | 0.6742 | 0.6530 | -0.0206 |
| ndcg_at_k | 0.6917 | 0.6899 | -0.0013 |
| p50 latency ms | 922.37 | 1788.07 | +870.22 |
| p95 latency ms | 1128.58 | 2608.10 | +540.21 |

## Interpretation

Hybrid RRF improves Hit@k and Recall@k over the dense top-k 5 baseline, but trails the dense top-k 10 candidate on this SciFact run. It costs substantially more latency because it combines dense search with sparse scoring.
