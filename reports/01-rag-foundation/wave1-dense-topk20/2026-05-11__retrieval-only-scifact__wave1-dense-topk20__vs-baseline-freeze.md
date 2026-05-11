---
report_id: rpt-01-rag-foundation-wave1-dense-topk20-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: wave1-dense-topk20
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: wave1-dense-topk20
---

## Goal

Compare dense top-k 20 against the dense top-k 5 frozen baseline on full SciFact without LLM calls.

## Metrics

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.9000 | +0.1000 |
| precision_at_k | 0.1727 | 0.0513 | -0.1214 |
| recall_at_k | 0.7771 | 0.8977 | +0.1206 |
| mrr_at_k | 0.6742 | 0.6849 | +0.0113 |
| ndcg_at_k | 0.6917 | 0.7301 | +0.0389 |
| p50 latency ms | 922.37 | 919.28 | +1.43 |
| p95 latency ms | 1128.58 | 1105.39 | -962.50 |

## Interpretation

Dense top-k 20 gives the strongest recall-oriented result so far: Hit@k and Recall@k improve beyond top-k 10, while Precision@k drops as expected from wider retrieval.
