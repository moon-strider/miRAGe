# store-index-variants

This experiment varies intra-backend index settings while keeping backend and
embedding model fixed.

## Moving variables

- `store_index_variant_id`

## Fixed variables

- store backend: `qdrant` for the qdrant-only comparison
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Runtime status

- qdrant HNSW default and high-ef are executable
- faiss flat, ivfflat, and ivfpq are executable through the paired backend experiment
