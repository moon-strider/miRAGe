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


## Retrieval-only full-dataset results: SciFact, 2026-05-11

Report:
- `reports/01-rag-foundation/wave1-dense-topk3/2026-05-11__retrieval-only-scifact__wave1-dense-topk3__vs-baseline-freeze.md`

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| hit_at_k | 0.8000 | 0.7467 | -0.0533 |
| precision_at_k | 0.1727 | 0.2644 | +0.0917 |
| recall_at_k | 0.7771 | 0.7224 | -0.0547 |
| mrr_at_k | 0.6742 | 0.6611 | -0.0131 |
| ndcg_at_k | 0.6917 | 0.6669 | -0.0248 |
| p50 latency ms | 922.37 | 911.00 | -11.37 |
| p95 latency ms | 1128.58 | 1204.35 | +75.77 |

Interpretation:
- No LLM calls were made; this is retrieval-only evaluation.
- Dense top-k 3 improves Precision@k but loses Hit@k, Recall@k, MRR@k, and NDCG@k. The baseline top-k 5 is not obviously over-retrieving on SciFact.
