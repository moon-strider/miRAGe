# wave1-tool-context-expansion

This experiment turns on a bounded second retrieval pass that expands context depth after the initial retrieval.

## Why this experiment exists

The repository now supports a minimal bounded tool-policy family.
This is not full agentic RAG, but it is enough to test a narrow and important question:
can a cheap second retrieval pass recover useful missing evidence without changing the rest of the stack?

This is worth testing because recent RAG best-practice work repeatedly points to a simple pattern:
- first retrieval often misses a needed supporting chunk
- blindly increasing baseline top-k can add noise everywhere
- a bounded follow-up retrieval can sometimes improve coverage only when needed

`tool-context-expansion-v1` is the safest policy in the current family because it changes retrieval depth without also changing the user query.

## What changes relative to baseline

- tool policy: `none` -> `tool-context-expansion-v1`

Everything else remains frozen to the baseline.

## Hypothesis

Expected direction:
- Hit@k and Recall@k may improve on questions where one-pass retrieval misses secondary evidence
- latency should increase because a second retrieval pass is always performed
- answer quality may improve only if the second pass contributes distinct evidence rather than duplicates
- the policy may be neutral or negative on already-easy questions because it adds cost without adding information

## Why this is worth doing early

This isolates the value of bounded orchestration before building anything more agentic.
If this simple policy already pays for itself, then future multi-step retrieval work is justified.
If it does not, then the stack probably needs stronger retrieval quality before it needs more control flow.

## Expected result template

Add after execution:
- one-pass vs bounded-two-pass retrieval/generation deltas
- extra latency and cost
- whether the second pass added distinct supporting chunks
- whether gains concentrated in only a few failure cases
