# wave1-dense-mmr

This retrieval-only experiment switches dense retrieval from plain top-k ranking to dense MMR diversification.

## Why this experiment exists

Dense top-k retrieval can return near-duplicate chunks when multiple chunks encode similar evidence.
MMR is a lightweight retrieval-stage diversification method that tries to balance query relevance against redundancy.
It is useful to test before any LLM stage because its effect is measurable directly through retrieval metrics and retrieved evidence diversity.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-dense-mmr-topk10-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- Hit@k and Recall@k may improve if baseline top-k contains redundant evidence
- MRR may stay flat or decrease if diversification moves the first relevant document lower
- NDCG should reveal whether diversified evidence is still well ranked
- retrieval latency should increase modestly because MMR reranks a wider dense candidate pool

## Why this is worth doing early

This is a no-LLM retrieval ablation with a clear interpretation: if MMR helps, later generation runs should receive less redundant context; if it hurts, the corpus likely benefits more from precision than diversity.

## Expected result template

Add after execution:
- measured retrieval metric deltas
- measured retrieval latency deltas
- whether MMR improved useful diversity or only disrupted ranking
