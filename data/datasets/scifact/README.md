# SciFact

## Purpose

Scientific claim verification retrieval benchmark for RAG runs that need short-to-medium evidence passages and high semantic precision.

## Why it is in this repository

This dataset covers a realistic retrieval regime where the question is closer to a claim than to a keyword query. It is useful for checking whether a system can recover semantically aligned scientific evidence instead of merely matching surface forms.

## Upstream sources

- Hugging Face dataset page: https://huggingface.co/datasets/BeIR/scifact
- BEIR direct archive: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

## Source snapshot pinned for this repository

- source_family: `beir`
- upstream_dataset_id: `BeIR/scifact`
- huggingface_sha: `b3b5335604bf5ee3c4447671af975ea25143d4f5`
- beir_archive_md5: `5f7d1de60b170fc8027bb7898e2efca1`

## Size and shape

- corpus documents: 5,183
- queries: 1,109
- total dataset rows on Hugging Face: 6,292
- Hugging Face total file size: 4.55 MB
- direct BEIR archive size: 2,816,079 bytes
- task type: scientific fact-checking retrieval
- language: English
- expected document profile: short and medium scientific abstracts/passages

## Retrieval characteristics

- primary stress: semantic evidence retrieval
- secondary stress: citation-style grounded answering
- weak fit for: exact numeric extraction stress tests

## Local directory contract

- `downloads/` — original archive fetched from upstream
- `raw/` — unpacked upstream files used by the runtime adapter
- `source.json` — machine-readable pinned metadata for automation

## Runtime adapter contract

At runtime, the SciFact adapter reads upstream files from `raw/scifact/` and maps them directly into repository `Document` and `EvalExample` records without writing repository-local normalized copies.

For SciFact, the adapter preserves claim text as the question and maps relevant evidence document ids into `gold_doc_ids`. It also preserves support/contradict labels in eval metadata for future richer metrics.
