# miRAGe

Minimal platform for controlled RAG experiments.

## Layout

- `config/` — global defaults and registries.
- `datasets/` — smoke fixtures and external benchmark source folders.
- `studies/` — executable study definitions and human results.
- `artifacts/` — ignored machine outputs and caches.
- `src/mirage/` — runtime code.
- `tests/` — regression tests.

## Experiment decision log

- 2026-05-11: froze the initial retrieval-only baseline on full SciFact: `text-embedding-3-small`, token chunks `1024/128`, Qdrant HNSW, dense top-k5, no reranker, no tool policy, no LLM generation. Details: `studies/rag-foundation/README.md` and `studies/rag-foundation/study.toml`.
- 2026-05-11: completed the first search/backend wave. Dense top-k10/top-k20 improved recall, BM25 and hybrid RRF did not beat dense search on SciFact, and FAISS flat matched Qdrant quality, so the vector store was not the quality bottleneck. Details: `studies/rag-foundation/README.md`.
- 2026-05-12: completed the embedding sweep and promoted `gemini-embedding-001` as the current retrieval baseline. It improved Recall@k from 0.7771 to 0.9509 and NDCG@k from 0.6912 to 0.8869 versus `text-embedding-3-small`. Details: `studies/rag-foundation/README.md`.
- 2026-05-12: next planned block is LLM-free macro chunking on the Gemini baseline. Prompting, generation, and agentic/tool experiments stay out of scope until retrieval stabilizes because LLM calls make runs much slower.

## Current retrieval baseline

`gemini-embedding-001` + full SciFact + `prep-basic-clean-v1` + token chunks `1024/128` + Qdrant HNSW cosine + dense top-k5 + no reranker + no tool policy + no LLM generation.

## Next LLM-free macro cases

Keep the current Gemini baseline frozen and compare only chunking strategy/scale:

1. control: token chunks `1024/128`.
2. small deterministic chunks: token chunks `512/64`.
3. large deterministic chunks: token chunks `2048/256`.
4. sentence-aware chunks: sentence chunks `1024/128`.
5. one semantic-chunking preset, without threshold grid search.

Do not run prompting, generation, or agentic/tool-policy experiments in this block.

## Quickstart

```bash
cp .env.example .env
uv sync --dev
docker compose up -d
just resolve
```
