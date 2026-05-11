---
report_id: rpt-01-rag-foundation-wave1-faiss-flat-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: wave1-faiss-flat
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: wave1-faiss-flat
---

## Goal

Compare local FAISS flat exact inner-product retrieval against the Qdrant HNSW frozen baseline on full SciFact retrieval metrics without LLM generation.

## Changed axis

- store backend: `qdrant` -> `faiss-local`
- store index: `idx-qdrant-hnsw-cosine-default-v1` -> `idx-faiss-flat-ip-v1`

## Metrics

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.8000 | +0.0000 |
| precision_at_k | 0.1727 | 0.1727 | +0.0000 |
| recall_at_k | 0.7771 | 0.7771 | +0.0000 |
| mrr_at_k | 0.6742 | 0.6742 | +0.0000 |
| ndcg_at_k | 0.6917 | 0.6917 | +0.0000 |
| p50 latency ms | 922.37 | 961.00 | +38.63 |
| p95 latency ms | 1128.58 | 1342.33 | +213.75 |

## Interpretation

FAISS flat exactly matches Qdrant HNSW retrieval quality on full SciFact for this configuration: Hit@k, Precision@k, Recall@k, MRR@k, and NDCG@k are unchanged. Latency is slightly worse in this run, with p50 increasing by +38.63 ms and p95 by +213.75 ms. This means backend/index choice is not a quality confounder for the current SciFact dense top-k5 baseline, but FAISS flat is not a latency win in this implementation.
