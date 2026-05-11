# inference-search

This experiment varies query-time search configuration.

## Moving variables

- `search_algorithm_id` (`dense`, `sparse`, `hybrid-rrf`, `dense-mmr`)

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`
