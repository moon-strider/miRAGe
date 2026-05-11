# inference-prompting

This experiment varies prompt protocol while leaving retrieval and storage fixed.

## Moving variables

- `prompt_variant_id`
  - `prompt-grounded-citations-v1`: default grounded answer with citations
  - `prompt-brief-grounded-v1`: same grounding constraints with brevity bias
  - `prompt-evidence-first-v1`: forces an explicit evidence section before the answer
  - `prompt-strict-abstain-v1`: maximizes abstention strictness when support is weak

## Fixed variables

- preprocessing: `prep-basic-clean-v1`
- chunking: `chunk-token-500-50-v1`
- chunking model: `none`
- store embedding: `emb-bge-small-en-v1.5`
- query embedding: `emb-bge-small-en-v1.5`
- search algorithm: `search-dense-topk5-v1`
