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
