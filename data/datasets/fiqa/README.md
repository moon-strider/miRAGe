# FiQA-2018

## Purpose

Financial question-answering retrieval benchmark for RAG runs that need domain jargon, practical user queries, and a mixture of semantic lookup with exact factual retrieval.

## Why it is in this repository

This dataset gives a more real-world user-query shape than scientific claim datasets. It is useful for testing financial terminology, paraphrase-heavy retrieval, and cases where the answer often depends on a specific passage rather than on broad topic matching.

## Upstream sources

- Hugging Face dataset page: https://huggingface.co/datasets/BeIR/fiqa
- BEIR direct archive: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip

## Source snapshot pinned for this repository

- source_family: `beir`
- upstream_dataset_id: `BeIR/fiqa`
- huggingface_sha: `979c07a7cb5ccc6ca009792241fa1250b98055dd`
- beir_archive_md5: `17918ed23cd04fb15047f73e6c3bd9d9`

## Size and shape

- corpus documents: 57,638
- queries: 6,648
- Hugging Face corpus bytes: 27,700,817
- Hugging Face query bytes: 321,680
- direct BEIR archive size: 17,948,027 bytes
- approximate tracked payload size from Hugging Face split metadata: 28.0 MB
- task type: financial article retrieval for question answering
- language: English
- expected document profile: forum-style and article-style passages with finance jargon

## Retrieval characteristics

- primary stress: mixed semantic and exact factual retrieval
- secondary stress: domain terminology and paraphrase robustness
- useful for: questions asking for practical financial interpretation, definitions, and concrete details

## Local directory contract

- `downloads/` — original archive fetched from upstream
- `raw/` — unpacked upstream files used by the runtime adapter
- `source.json` — machine-readable pinned metadata for automation

## Runtime adapter contract

At runtime, the FiQA adapter reads upstream files from `raw/fiqa/` and maps them directly into repository `Document` and `EvalExample` records without writing repository-local normalized copies.

For FiQA, evaluation rows preserve the original user query and map relevant document ids into `gold_doc_ids`. Gold free-form answers are not available in the BEIR package, so the current adapter marks FiQA eval rows as retrieval-focused via metadata.
