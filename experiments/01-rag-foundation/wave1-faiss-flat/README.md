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

## Retrieval-only full-dataset results: SciFact, 2026-05-11

Report:
- `reports/01-rag-foundation/wave1-faiss-flat/2026-05-11__retrieval-only-scifact__wave1-faiss-flat__vs-baseline-freeze.md`

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.8000 | +0.0000 |
| precision_at_k | 0.1727 | 0.1727 | +0.0000 |
| recall_at_k | 0.7771 | 0.7771 | +0.0000 |
| mrr_at_k | 0.6742 | 0.6742 | +0.0000 |
| ndcg_at_k | 0.6917 | 0.6917 | +0.0000 |
| p50 latency ms | 922.37 | 961.00 | +38.63 |
| p95 latency ms | 1128.58 | 1342.33 | +213.75 |

Interpretation:
- No LLM calls were made; this is retrieval-only evaluation.
- FAISS flat matches Qdrant HNSW on all retrieval quality metrics for the frozen dense top-k5 setup.
- The backend is not a quality confounder for this SciFact baseline.
- Runtime latency is slightly worse than Qdrant in this implementation, so FAISS flat should remain a correctness/control baseline rather than become the default.
