# Report Schema

## Purpose

This document defines the required structure for every human-readable report in
`reports/`.

A report is a uniform summary of measured results.
It links to machine-readable artifacts under `runs/`.
It does not prescribe future actions.

## Directory layout

Reports are grouped by experiment group and experiment id.

Required directory shape:

```text
reports/
  SCHEMA.md
  <group_id>/
    <experiment_id>/
      <date>__<candidate-id>__vs-<baseline-id>.md
```

The report file naming rule is:

`<date>__<candidate-id>__vs-<baseline-id>.md`

## Metadata block

Every report must start with a YAML metadata block.

Required fields:

```yaml
report_id: rpt-01-rag-foundation-emb-large-vs-bge-v1
created_at: 2025-02-10T14:00:00Z
group_id: 01-rag-foundation
experiment_id: store-embedding-models
baseline_id: emb-bge-small-en-v1.5
candidate_id: emb-text-embedding-3-large
dataset_id: ds-docs-v1
evalset_id: ev-dev-v1
preprocessing_variant_id: prep-basic-clean-v1
chunking_variant_id: chunk-token-500-50-v1
chunking_model_id: none
load_variant_id: load-prep-basic-clean-v1__chunk-token-500-50-v1__none
store_backend_id: qdrant
store_index_variant_id: idx-qdrant-hnsw-cosine-default-v1
store_embedding_model_id: emb-text-embedding-3-large
query_embedding_model_id: emb-text-embedding-3-large
search_algorithm_id: search-dense-topk5-v1
reranker_id: none
prompt_variant_id: prompt-grounded-citations-v1
tool_policy_id: none
inference_variant_id: inf-emb-text-embedding-3-large__search-dense-topk5-v1__none__prompt-grounded-citations-v1__none
generation_model_ids:
  - gen-llama-3.1-8b
  - gen-grok-4-fast
  - gen-minimax-2.7
run_paths:
  - runs/answers/...
  - runs/answers/...
price_catalog_version: openrouter-public-v1
```

If the report covers a single generation model, keep `generation_model_ids` as a one-item list.

## Mandatory sections

Every report must contain the following sections in this exact order:

1. `## Goal`
2. `## Compared variants`
3. `## Artifact reuse`
4. `## Metrics summary`
5. `## Cost summary`
6. `## Failure cases`
7. `## Conclusion`

## Required content by section

### Goal

State the question answered by the report.
Keep it short and factual.

### Compared variants

Include a compact table showing what changed and what stayed fixed.

Minimum table shape:

| Axis | Baseline | Candidate | Status |
| --- | --- | --- | --- |
| preprocessing | ... | ... | changed or fixed |
| chunking | ... | ... | changed or fixed |
| chunking model | ... | ... | changed or fixed |
| store backend | ... | ... | changed or fixed |
| index variant | ... | ... | changed or fixed |
| store embedding | ... | ... | changed or fixed |
| query embedding | ... | ... | changed or fixed |
| search algorithm | ... | ... | changed or fixed |
| reranker | ... | ... | changed or fixed |
| prompt variant | ... | ... | changed or fixed |
| tool policy | ... | ... | changed or fixed |
| generation models | ... | ... | changed or fixed |

### Artifact reuse

State exactly which machine-readable artifacts were reused and which were rebuilt.

Minimum table shape:

| Layer | Artifact id or path | Reused or rebuilt | Why |
| --- | --- | --- | --- |
| prepared | ... | reused or rebuilt | ... |
| chunks | ... | reused or rebuilt | ... |
| store | ... | reused or rebuilt | ... |
| retrieval | ... | reused or rebuilt | ... |
| answers | ... | reused or rebuilt | ... |

### Metrics summary

Metrics must be shown per generation model.

Minimum table shape:

| generation_model_id | Hit@k | Recall@k | MRR@k | Exact Match | Token F1 | Citation Hit Rate | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Add storage footprint and ingest time when the comparison changes store-time behavior.
Add ranking metrics when the comparison changes ranking behavior.

### Cost summary

Always separate one-time build cost from recurring query cost.

Minimum table shape:

| generation_model_id | build preprocessing | build chunking | build store embeddings | mean query cost over eval | projected query cost at 1M | amortized total at 1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Local phases must be recorded as `0` for external token cost.
Infrastructure cost, if tracked, must be shown in a separate table.

### Failure cases

Include three to five representative cases.

Required coverage:

- one strong success
- one typical middling case
- one retrieval failure
- one generation failure
- one surprising or ambiguous case when available

Each case must include:

- `qid`
- short question text
- expected answer summary
- retrieved document ids
- model answer summary
- why the case matters

### Conclusion

State the measured conclusion.
The conclusion must reference tradeoffs, not only a winner label.

## Linkage to run artifacts

Every report must link to the machine-readable artifacts it summarizes.
Do not inline raw JSONL dumps into the report.

At minimum, reference:

- `runs/.../resolved_spec.json`
- `runs/.../retrieval.jsonl`
- `runs/.../answers.jsonl`
- `runs/.../metrics.json`

## Comparison policy

A report is valid only if it compares a candidate against a named baseline.
The only exception is the first baseline freeze of a new experiment.

## Short authoring template

```md
---
report_id: rpt-...
created_at: ...
group_id: ...
experiment_id: ...
baseline_id: ...
candidate_id: ...
dataset_id: ...
evalset_id: ...
preprocessing_variant_id: ...
chunking_variant_id: ...
chunking_model_id: ...
load_variant_id: ...
store_backend_id: ...
store_index_variant_id: ...
store_embedding_model_id: ...
query_embedding_model_id: ...
search_algorithm_id: ...
reranker_id: ...
prompt_variant_id: ...
tool_policy_id: ...
inference_variant_id: ...
generation_model_ids:
  - ...
run_paths:
  - ...
price_catalog_version: openrouter-public-v1
---

## Goal

## Compared variants

## Artifact reuse

## Metrics summary

## Cost summary

## Failure cases

## Conclusion
```
