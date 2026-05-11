---
report_id: rpt-01-rag-foundation-baseline-freeze-retrieval-only-scifact
created_at: 2026-05-11
mode: retrieval-only
group_id: 01-rag-foundation
experiment_id: baseline-freeze
dataset_id: ds-beir-scifact-v1
evalset_id: ev-beir-scifact-v1
baseline_id: baseline-freeze
candidate_id: baseline-freeze
---

## Goal

Record full SciFact retrieval-only baseline metrics without any LLM generation calls.

## Configuration

- dataset: `ds-beir-scifact-v1`
- eval set: `ev-beir-scifact-v1`
- search algorithm: `search-dense-topk5-v1`
- reranker: `none`
- tool policy: `none`
- LLM calls: none

## Metrics

| examples | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms | mean query embedding cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0.8000 | 0.1727 | 0.7771 | 0.6736 | 0.6912 | 917.85 | 2067.89 | $0.000000 |

## Interpretation

This is the full-dataset retrieval baseline for wave1. It intentionally excludes answer generation and therefore reports only retrieval, latency, and embedding-cost metrics.
