---
report_id: rpt-01-rag-foundation-wave1-dense-topk3-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: wave1-dense-topk3
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: wave1-dense-topk3
---

## Goal

Compare dense top-k 3 against the dense top-k 5 frozen baseline on full SciFact without LLM calls.

## Metrics

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.7467 | -0.0533 |
| precision_at_k | 0.1727 | 0.2644 | +0.0917 |
| recall_at_k | 0.7771 | 0.7224 | -0.0547 |
| mrr_at_k | 0.6742 | 0.6611 | -0.0131 |
| ndcg_at_k | 0.6917 | 0.6669 | -0.0248 |
| p50 latency ms | 922.37 | 911.00 | -11.37 |
| p95 latency ms | 1128.58 | 1204.35 | +75.77 |

## Interpretation

Dense top-k 3 improves Precision@k but loses Hit@k, Recall@k, MRR@k, and NDCG@k. The baseline top-k 5 is not obviously over-retrieving on SciFact.
