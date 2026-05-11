# wave1-dense-topk3

This retrieval-only experiment narrows dense retrieval from top-k 5 to top-k 3.

## Why this experiment exists

A narrow top-k test shows how much retrieval quality depends on the last two baseline candidates.
It is the cheapest way to quantify the precision/recall tradeoff in the opposite direction from top-k 10 and top-k 20.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-dense-topk3-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- Precision@k may improve because fewer candidates are returned
- Hit@k and Recall@k may drop if relevant evidence often appears at ranks 4-5
- latency should improve slightly

## Why this is worth doing early

It tells whether the baseline is already over-retrieving. If top-k 3 holds quality, later LLM stages can use less context and lower cost.

## Expected result template

Add after execution:
- measured retrieval metric deltas
- latency deltas
- whether top-k 3 is a viable cheaper retrieval setting
