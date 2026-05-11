# wave1-faiss-flat

This retrieval-only experiment switches the vector backend from Qdrant HNSW to local FAISS flat inner-product search.

## Why this experiment exists

A backend comparison can be evaluated without LLM generation because retrieval metrics directly show whether backend/index behavior changes candidate evidence.
FAISS flat search is an exact local baseline and is useful for checking whether approximate serving infrastructure affects retrieval quality.

## What changes relative to baseline

- store backend: `qdrant` -> `faiss-local`
- store index: `idx-qdrant-hnsw-cosine-default-v1` -> `idx-faiss-flat-ip-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- retrieval quality should be similar if Qdrant HNSW is already accurate on this corpus
- latency may differ because FAISS runs locally without the Qdrant HTTP service
- large quality differences would suggest backend/index semantics need normalization before fair comparisons

## Why this is worth doing early

It tests whether backend choice itself is a confounder before later retrieval experiments are interpreted.

## Expected result template

Add after execution:
- measured retrieval metric deltas
- latency deltas
- whether FAISS can be treated as a comparable local backend
