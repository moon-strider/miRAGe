# wave1-dense-topk10

This experiment increases first-stage dense retrieval depth from the frozen baseline `search-dense-topk5-v1` to `search-dense-topk10-v1`.

## Why this experiment exists

This is the cleanest way to test whether the baseline is recall-limited.
The literature repeatedly describes a top-k trade-off:
- larger retrieval sets improve the chance that relevant evidence is present
- but larger sets can also dilute the prompt with weaker context
- several RAG best-practice discussions treat retrieval depth as one of the first ablations to run because it changes recall without changing the retrieval family itself

That makes this the lowest-risk and most interpretable first perturbation of the baseline.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-dense-topk10-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- retrieval recall metrics should improve or stay flat
- Hit@k should not decrease
- Recall@k should improve more often than Exact Match
- latency and prompt cost should increase modestly
- generation quality may improve only slightly if the extra retrieved chunks are mostly redundant

## Why this is worth doing early

If this experiment wins cleanly, the baseline was under-retrieving and many later experiments should probably compare against top-k 10 rather than top-k 5.
If it does not win, that is strong evidence that retrieval depth is already near the useful range and that precision-oriented changes are more valuable than simply widening recall.

## Expected result template

Add after execution:
- measured retrieval deltas
- measured generation deltas
- measured latency/cost deltas
- whether the extra five chunks were useful evidence or mostly noise
