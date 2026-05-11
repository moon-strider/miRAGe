# wave1-semantic-chunking

This experiment replaces the token baseline with a tuned semantic chunking configuration.

## Why this experiment exists

Chunking is one of the most important upstream RAG choices.
The current baseline uses token chunking because it is the least assumption-heavy reference point, but semantic chunking is the most plausible next step when the goal is to improve evidence boundaries rather than only retrieval depth.

Recent best-practice literature and engineering writeups consistently describe the same trade-off:
- semantic chunking can align chunk boundaries better with meaning
- better boundaries can improve retrieval precision and reduce mixed-topic chunks
- but semantic chunking costs more at ingest time and can be unstable if thresholds are poorly chosen

This experiment uses a conservative semantic configuration instead of the full semantic matrix because the first question is not "what is the perfect threshold?" but "does semantic chunking beat the plain token baseline at all on this stack?"

## What changes relative to baseline

- load prototype: token -> semantic
- chunking: `chunk-token-1024-128-v1` -> `chunk-semantic-v1`
- chunking model: `none` -> `emb-text-embedding-3-small`
- semantic similarity threshold: `0.85`
- semantic minimum sentences per chunk: `2`

Everything else remains frozen to the baseline.

## Why these semantic settings were chosen

- `emb-text-embedding-3-small`: keeps the semantic chunker aligned with the baseline embedding family instead of introducing a totally different provider shape
- threshold `0.85`: intentionally conservative, favoring cleaner semantic splits over broad merging
- minimum sentences `2`: reduces fragmentation from a single noisy sentence

## Hypothesis

Expected direction:
- retrieval precision, MRR, and citation quality may improve if chunk boundaries become cleaner
- total chunk count may shift in either direction depending on document structure
- ingest cost should increase because semantic chunking requires sentence embeddings
- answer quality may improve only if the corpus suffers from token-boundary mixing in the baseline

## Why this is worth doing early

This is the cleanest upstream structural experiment in the first wave.
If it wins, chunk quality deserves more investment.
If it does not, we should avoid overspending on sophisticated chunking before exhausting cheaper retrieval-stage improvements.

## Expected result template

Add after execution:
- chunk-count change vs baseline
- retrieval/generation deltas
- ingest-cost delta
- whether semantic chunking improved evidence cleanliness or merely changed artifact shape
