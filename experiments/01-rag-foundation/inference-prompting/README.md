# inference-prompting

This experiment varies prompt protocol while leaving retrieval and storage fixed.

## Moving variables

- `prompt_variant_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`
- search algorithm: `search-dense-topk5-v1`
