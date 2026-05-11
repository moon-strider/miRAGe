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

## Actual results: 2026-05-11

Report:
- `reports/01-rag-foundation/wave1-dense-topk10/2026-05-11__wave1-dense-topk10__vs-baseline-freeze.md`

| generation_model_id | Hit@k delta | Precision@k delta | Recall@k delta | MRR@k delta | NDCG@k delta | Exact Match delta | Token F1 delta | Citation Hit Rate delta | p50 delta ms | p95 delta ms | mean query cost delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gen-grok-4-fast` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -37.82 | +14243.03 | `-$0.000001` |
| `gen-llama-3.1-8b` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -94.95 | -508.44 | `$0.000000` |
| `gen-minimax-2.7` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +920.65 | +131.09 | `$0.000000` |

Interpretation:
- The recall hypothesis did not produce a measurable gain on this smoke fixture: baseline retrieval was already saturated at `1.0000` for Hit@k, Recall@k, MRR@k, and NDCG@k.
- Precision@k stayed `0.2000` in the report because only five chunks exist in this tiny corpus; widening top-k cannot add more than the whole corpus here.
- Generation quality did not move: Exact Match, Token F1, and Citation Hit Rate are unchanged for all three generation models.
- Latency was mostly noise-dominated on this tiny run; `gen-grok-4-fast` had one large p95 outlier, while `gen-llama-3.1-8b` was slightly faster than baseline.
- This experiment does not justify changing the baseline from top-k 5 to top-k 10 for the smoke dataset. It remains worth retesting on larger benchmark datasets where top-k 10 can actually retrieve a larger candidate set.
