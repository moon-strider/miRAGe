# load-semantic-chunking

This experiment varies the model used for semantic chunking.

## Moving variables

- `chunking_model_id`
- `generation_model_id`

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking variant: `chunk-semantic-v1`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Matrix

- semantic chunker model: `emb-bge-small-en-v1.5`, `emb-text-embedding-3-small`, `emb-text-embedding-3-large`
- generation model: all three fixed generation models

## Runtime status

- executable in the current runtime
- semantic boundaries are determined by sentence-level similarity thresholding
- chunking artifacts remain model-bound through `chunking_model_id`

The runtime treats `chunking_model_id` as a load-time variable and
`generation_model_id` as a separate inference-time variable.
