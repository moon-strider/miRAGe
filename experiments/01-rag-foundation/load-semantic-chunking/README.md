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

- semantic chunker model: all three fixed generation models
- generation model: all three fixed generation models

## Runtime status

Semantic chunking is scaffolded in the experiment system but is not implemented in runtime yet.

The runtime treats `chunking_model_id` as a load-time variable and
`generation_model_id` as a separate inference-time variable.
