# wave1-hybrid-rrf

This experiment switches the retrieval family from dense-only search to hybrid BM25+dense retrieval with reciprocal-rank fusion.

## Why this experiment exists

Hybrid retrieval is one of the most consistent recommendations in both retrieval literature and production RAG practice.
Across practitioner summaries and benchmark discussions, the same pattern appears:
- dense retrieval captures paraphrase and semantic similarity
- BM25 catches exact identifiers, literal constraints, and rare terms
- the two methods fail differently
- RRF is a robust default because it avoids score-calibration problems between sparse and dense systems

That makes hybrid retrieval the most important family-level comparison against a dense baseline.

## What changes relative to baseline

- search algorithm: `search-dense-topk5-v1` -> `search-hybrid-rrf-topk10-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- retrieval recall metrics should improve if lexical cues matter in the evaluation set
- MRR/NDCG may improve if BM25 recovers chunks that dense retrieval misses entirely
- latency should increase because two candidate generators run before fusion
- answer quality should improve only if the newly recovered lexical evidence is actually relevant

## Why this is worth doing early

If hybrid wins, it becomes a strong candidate for the default retrieval family in later waves.
If hybrid does not win on the baseline corpus, that tells us either the benchmark set is too semantically smooth or the dense baseline is already sufficient on this workload.
Either outcome is useful.

## Expected result template

Add after execution:
- dense vs hybrid retrieval metrics
- whether lexical rescue cases were visible in failure analysis
- fusion cost/latency impact
- whether hybrid improved recall only or also improved final answers
