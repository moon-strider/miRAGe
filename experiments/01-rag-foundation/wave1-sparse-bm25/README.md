# wave1-sparse-bm25

This retrieval-only experiment switches from dense retrieval to sparse BM25 lexical retrieval.

## Why this experiment exists

BM25 is the classic lexical baseline for retrieval benchmarks.
It is cheap, deterministic, and often strong on datasets where exact terminology matters.
Testing it separately from hybrid RRF shows whether lexical retrieval alone is competitive or whether dense retrieval is carrying most of the result.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-sparse-bm25-topk10-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- BM25 may improve exact-term factual queries
- semantic/paraphrase-heavy queries may regress versus dense retrieval
- latency may increase in the current simple runtime because sparse search scores local chunk text at query time

## Why this is worth doing early

This is a no-LLM retrieval baseline that helps interpret hybrid RRF. If sparse BM25 alone is weak but hybrid helps, fusion is rescuing specific lexical cases without replacing dense retrieval.

## Expected result template

Add after execution:
- measured retrieval metric deltas
- latency deltas
- whether BM25 is competitive alone or only useful as a hybrid component


## Retrieval-only full-dataset results: SciFact, 2026-05-11

Report:
- `reports/01-rag-foundation/wave1-sparse-bm25/2026-05-11__retrieval-only-scifact__wave1-sparse-bm25__vs-baseline-freeze.md`

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.6867 | -0.1133 |
| precision_at_k | 0.1727 | 0.0742 | -0.0985 |
| recall_at_k | 0.7771 | 0.6688 | -0.1083 |
| mrr_at_k | 0.6742 | 0.5095 | -0.1641 |
| ndcg_at_k | 0.6917 | 0.5438 | -0.1474 |
| p50 latency ms | 922.37 | 1722.37 | +804.52 |
| p95 latency ms | 1128.58 | 2005.89 | -62.00 |

Interpretation:
- No LLM calls were made; this is retrieval-only evaluation.
- Sparse BM25 underperforms the dense baseline on full SciFact in this runtime. It is not competitive alone, but it remains useful as a component to interpret hybrid RRF.
