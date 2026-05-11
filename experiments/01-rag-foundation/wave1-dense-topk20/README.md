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


## Retrieval-only full-dataset results: SciFact, 2026-05-11

Report:
- `reports/01-rag-foundation/wave1-dense-topk20/2026-05-11__retrieval-only-scifact__wave1-dense-topk20__vs-baseline-freeze.md`

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.9000 | +0.1000 |
| precision_at_k | 0.1727 | 0.0513 | -0.1214 |
| recall_at_k | 0.7771 | 0.8977 | +0.1206 |
| mrr_at_k | 0.6736 | 0.6849 | +0.0113 |
| ndcg_at_k | 0.6912 | 0.7301 | +0.0389 |
| p50 latency ms | 917.85 | 919.28 | +1.43 |
| p95 latency ms | 2067.89 | 1105.39 | -962.50 |

Interpretation:
- No LLM calls were made; this is retrieval-only evaluation.
- Dense top-k 20 gives the strongest recall-oriented result so far: Hit@k and Recall@k improve beyond top-k 10, while Precision@k drops as expected from wider retrieval.
- This is the best recall candidate so far and is a strong candidate for a later reranker/LLM stage.
