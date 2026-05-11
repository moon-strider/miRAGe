# load-semantic-chunking

This experiment varies the model used for semantic chunking.

## Moving variables

- `chunking_model_id`
- `semantic_similarity_threshold`
- `semantic_min_sentences_per_chunk`
- `generation_model_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking variant: `chunk-semantic-v1`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Matrix

- semantic chunker model: `emb-bge-small-en-v1.5`, `emb-text-embedding-3-small`, `emb-text-embedding-3-large`
- similarity threshold: `0.7`, `0.85`
- minimum sentences per chunk: `1`, `2`
- generation model: all three fixed generation models

## Runtime status

- executable in the current runtime
- semantic boundaries are determined by sentence-level similarity thresholding
- chunk size still caps overlong semantic groups
- lower thresholds should create fewer, broader chunks; higher thresholds should create finer-grained chunks
- larger `semantic_min_sentences_per_chunk` prevents fragmentation from single noisy sentences
- chunking artifacts remain model-bound through `chunking_model_id`

The runtime treats `chunking_model_id` as a load-time variable and
`generation_model_id` as a separate inference-time variable.
