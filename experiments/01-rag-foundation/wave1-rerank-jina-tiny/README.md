# wave1-rerank-jina-tiny

This experiment adds a lightweight cross-encoder reranker on top of a wider dense retrieval pool.

## Why this experiment exists

Reranking is the standard second-stage precision fix in modern RAG stacks.
The common pattern in both benchmarks and production playbooks is:
- retrieve a somewhat wider candidate set for recall
- rerank a smaller pool with a more precise cross-encoder
- keep only the best evidence for generation

This repository now has multiple rerankers, but `rerank-jina-tiny-v1` is the right first reranker experiment because it is still lightweight enough to remain practical while being meaningfully different from the no-reranker baseline.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-dense-topk10-v1`
- reranker: `none` -> `rerank-jina-tiny-v1`

This is an intentionally coupled change.
A reranker needs a larger candidate pool to be useful, so comparing reranker-on with top-k5 would understate its value.

## Hypothesis

Expected direction:
- MRR/NDCG should improve more than raw Hit@k
- Exact Match / Token F1 / citation quality may improve if the reranker moves the truly relevant chunk earlier in the prompt
- latency should increase materially
- query cost should stay roughly similar, but end-to-end runtime should rise because reranking is the extra expensive step

## Why this is worth doing early

This tests whether the project is precision-limited after first-stage retrieval.
If reranking wins clearly, then later retrieval-family experiments should be judged partly by how well they feed the reranker rather than only by first-stage retrieval metrics.

## Expected result template

Add after execution:
- baseline vs reranked ranking metrics
- whether answer quality improved along with retrieval ordering
- reranker latency overhead
- whether the larger candidate pool introduced useful recall or only extra cost
