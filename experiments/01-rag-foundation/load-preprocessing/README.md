# load-preprocessing

This experiment varies preprocessing while keeping chunking, storage, retrieval,
and prompting fixed.

## Moving variables

- `preprocessing_variant_id`
  - `prep-basic-clean-v1`: whitespace cleanup only
  - `prep-basic-clean-dedupe-v1`: cleanup plus duplicate document removal by normalized title+text
  - `prep-basic-clean-metadata-v1`: cleanup plus metadata/header injection into document text

## Fixed variables

- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Matrix

- preprocessing variants from the overlay
- generation model from the group
