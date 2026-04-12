# load-deterministic-chunking

This experiment varies deterministic chunking policies while keeping
preprocessing and store settings fixed.

## Moving variables

- `chunking_variant_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Runtime status

- executable in the current runtime
- token window variants use `768/128` and `1024/128`
- sentence variant uses sentence-aware packing at `1024/128`

## Matrix

- `chunk-token-1024-128-v1`
- `chunk-token-768-128-v1`
- `chunk-sentence-1024-128-v1`
- generation model from the group
