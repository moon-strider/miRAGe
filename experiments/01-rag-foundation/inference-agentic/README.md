# inference-agentic

This experiment varies tool-use policy or orchestration style while keeping the
rest of the retrieval stack fixed.

## Moving variables

- `tool_policy_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`
- prompt variant: `prompt-grounded-citations-v1`

Current runtime support is baseline-only. Non-`none` policies are scaffolded but
not yet implemented.
