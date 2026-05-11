# store-backends

This experiment varies the storage backend while keeping the load pipeline and
baseline embedding model fixed.

## Moving variables

- `store_backend_id`
- `store_index_variant_id`

## Coupled cases

Backends and index variants are paired through explicit cases.
Only cases whose backend runtime status is implemented can be executed.

## Runtime status

`qdrant` is active.
`faiss-local` is active for flat inner-product runtime and can be executed.
