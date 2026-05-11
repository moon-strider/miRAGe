# baseline-freeze

This experiment defines the executable benchmark baseline for the first comparison wave.

It is intentionally simple, cheap enough to rerun, and interpretable enough to serve as the reference point for nearby ablations.

## What this baseline freezes

- dataset: `ds-docs-v1`
- eval set: `ev-dev-v1`
- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-1024-128-v1`
- chunking model: `none`
- store backend: `qdrant`
- store index: `idx-qdrant-hnsw-cosine-default-v1`
- store embedding: `emb-text-embedding-3-small`
- query embedding: `emb-text-embedding-3-small`
- search algorithm: `search-dense-topk5-v1`
- reranker: `none`
- prompt: `prompt-grounded-citations-v1`
- tool policy: `none`
- generation model: all three pinned generation models from the group

## Why each choice is frozen this way

### Dataset = `ds-docs-v1`

This repository fixture is the smallest fully controlled executable corpus.
It is not meant to be the final scientific benchmark; it is the stable smoke-test baseline that keeps baseline ingest, eval, and reporting fast and deterministic while the broader experiment surface is still being built out.

### Eval set = `ev-dev-v1`

The paired dev fixture gives a compact, repeatable answer/retrieval target for regression testing.
The point of the baseline is not dataset realism yet; the point is to establish one reference configuration that always resolves and runs.

### Preprocessing = `prep-basic-clean-v1`

The baseline should avoid opinionated document reshaping.
Whitespace cleanup is the minimum reasonable normalization step.
It removes accidental formatting noise without changing evidence density, document count, or metadata exposure.
This makes later preprocessing comparisons easy to interpret.

### Chunking = `chunk-token-1024-128-v1`

This is the most conservative chunking choice in the repository:
- token-based chunking is deterministic
- 1024 tokens keeps evidence reasonably complete
- 128 overlap reduces boundary misses without creating extreme duplication

This is a better reference point than sentence or semantic chunking because it adds fewer modeling assumptions.

### Chunking model = `none`

A baseline should not depend on an extra semantic segmentation model unless semantic segmentation itself is the thing being studied.
Freezing this to `none` keeps load-time artifacts cheaper and easier to compare.

### Store backend = `qdrant`

Qdrant is the default vector store in the repository and the most production-like baseline backend here.
It provides a stable online retrieval path and avoids turning the baseline into a backend bakeoff.
FAISS remains valuable for dedicated backend/index experiments, but not as the default reference point.

### Store index = `idx-qdrant-hnsw-cosine-default-v1`

Default HNSW is the plainest executable qdrant index choice.
It is the right baseline because it is neither aggressively tuned nor intentionally degraded.
That makes later index changes interpretable as real deltas from a normal serving setup.

### Store embedding = `emb-text-embedding-3-small`

This is the repository's default executable baseline embedding.
It is a stronger practical baseline than the smallest local option while still being much cheaper than the larger hosted model.
Using the same model for indexing and querying avoids confounding the baseline with embedding mismatch.

### Query embedding = `emb-text-embedding-3-small`

The query encoder stays matched to the store encoder.
That removes one extra degree of freedom from the baseline and makes retrieval behavior easier to reason about.

### Search algorithm = `search-dense-topk5-v1`

Dense top-k retrieval is the simplest first-stage retrieval baseline.
`top_k = 5` is intentionally modest:
- enough evidence for a small grounded answer
- low latency
- low prompt bloat
- low dilution from irrelevant chunks

It gives a clean contrast against top-k expansion, hybrid retrieval, and MMR/diversification experiments.

### Reranker = `none`

The baseline should measure first-stage retrieval directly.
If reranking is enabled by default, then search experiments and reranker experiments become harder to disentangle.
Leaving reranking off makes reranker gains visible when they are introduced.

### Prompt = `prompt-grounded-citations-v1`

This is the clearest grounded-answer prompt in the current prompt family:
- requires using only provided context
- enforces explicit insufficient-context behavior
- requires citations

It is the least ambiguous prompt baseline for RAG evaluation.

### Tool policy = `none`

A baseline should be single-pass and bounded.
Agentic behavior is interesting only when compared against a simple non-agentic reference.
Keeping tool policy at `none` ensures that later tool-policy experiments measure extra retrieval/control flow rather than hidden baseline complexity.

### Generation models = pinned group trio

The baseline resolves to three concrete specs, one per pinned generation model:
- `gen-llama-3.1-8b`
- `gen-grok-4-fast`
- `gen-minimax-2.7`

This keeps the baseline aligned with the group's main comparison contract while holding everything else fixed.

## Output shape

The experiment resolves to three concrete specs, one per generation model.
All upstream load/store/retrieval settings remain frozen.
Only the generation model changes across the resolved baseline specs.


## Retrieval-only full-dataset results: SciFact, 2026-05-11

Report:
- `reports/01-rag-foundation/baseline-freeze/2026-05-11__retrieval-only-scifact__baseline-freeze.md`

| dataset | examples | Hit@k | Precision@k | Recall@k | MRR@k | NDCG@k | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ds-beir-scifact-v1` | 300 | 0.8000 | 0.1727 | 0.7771 | 0.6736 | 0.6912 | 917.85 | 2067.89 |

Interpretation:
- This result is retrieval-only: no LLM generation calls were made.
- The baseline retrieves at least one gold document for 80.00% of SciFact test examples.
- Recall@k is 0.7771 and NDCG@k is 0.6912, so there is enough headroom for retrieval-side experiments to be meaningful.
