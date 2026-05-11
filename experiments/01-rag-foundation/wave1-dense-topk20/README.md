# wave1-dense-topk20

This retrieval-only experiment widens dense retrieval from the baseline top-k 5 to top-k 20.

## Why this experiment exists

The top-k 10 run tests a moderate recall increase. Top-k 20 tests whether there is still useful recall headroom or whether wider dense retrieval mostly adds noise.
This is cheap to evaluate without LLM calls because only retrieval metrics are computed.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-dense-topk20-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- Hit@k and Recall@k should improve over baseline
- Precision@k should drop because more candidates are returned
- MRR may improve slightly or stay flat depending on whether new relevant documents appear near the top
- latency should increase moderately

## Why this is worth doing early

It identifies the useful dense-retrieval recall ceiling before adding rerankers or LLM answer generation.

## Expected result template

Add after execution:
- measured retrieval metric deltas
- latency deltas
- whether top-k 20 gives enough extra recall to justify later reranking
