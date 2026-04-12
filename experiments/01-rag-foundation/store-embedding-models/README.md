# store-embedding-models

This experiment compares document/query embedding choices as store-time and
query-time variables.

## Moving variables

- `store_embedding_model_id`
- `query_embedding_model_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`

## Coupled cases

The overlay uses explicit cases so that store and query embedding ids move
in lockstep by default.
