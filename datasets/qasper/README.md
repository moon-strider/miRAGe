# Qasper

## Purpose

Long-document question answering benchmark over NLP research papers for RAG evaluations that need section-aware retrieval, long-context chunking, and evidence grounding across full documents.

## Why it is in this repository

This dataset covers the long-document regime that the tiny smoke fixture cannot represent. It is useful for testing chunking policies, retrieval under long context, and answer grounding when evidence can be distributed across sections of a paper.

## Upstream sources

- Hugging Face dataset page: https://huggingface.co/datasets/allenai/qasper
- Official train/dev archive: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
- Official test archive: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz

## Source snapshot pinned for this repository

- source_family: `qasper`
- upstream_dataset_id: `allenai/qasper`
- huggingface_sha: `fdc9d8214fbab5dd782958601db4d678e6934a54`
- license: `CC BY 4.0`

## Size and shape

- papers: 1,585
- total questions: 5,049
- train papers: 888
- validation papers: 281
- test papers: 416
- train split bytes: 27,277,970
- validation split bytes: 9,535,330
- test split bytes: 9,535,330
- total download size: 14,700,917 bytes
- total dataset size after processing metadata: 36,813,300 bytes
- total size in bytes reported by Hugging Face API: 68,556,447
- task type: long-document evidence-based question answering
- language: English
- expected document profile: full research papers with sections and paragraph structure

## Retrieval characteristics

- primary stress: long-document retrieval and chunking quality
- secondary stress: evidence grounding across multiple sections
- useful for: semantic retrieval over long contexts and answers that need paragraph-level support

## Local directory contract

- `downloads/` — original archives fetched from upstream
- `raw/` — unpacked upstream files used by the runtime adapter
- `source.json` — machine-readable pinned metadata for automation

## Runtime adapter contract

At runtime, the Qasper adapter reads upstream files from `raw/` and maps them directly into repository `Document` and `EvalExample` records without writing repository-local normalized copies.

The adapter concatenates abstract and ordered sections into each `Document`, preserves section names in metadata, maps each question to the owning paper id in `gold_doc_ids`, and preserves evidence payloads in eval metadata for future finer-grained evaluation.
