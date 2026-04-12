# baseline-freeze

This experiment defines the zero point for the first large experiment group.

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`

## Matrix

- generation model: all three fixed generation models from the group

## Output shape

The experiment resolves to three concrete specs, one per generation model.
