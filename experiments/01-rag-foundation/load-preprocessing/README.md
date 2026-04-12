# load-preprocessing

This experiment varies preprocessing while keeping chunking, storage, retrieval,
and prompting fixed.

## Moving variables

- `preprocessing_variant_id`

## Fixed variables

- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Matrix

- preprocessing variants from the overlay
- generation model from the group
