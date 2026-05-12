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
- 2026-05-12: completed the LLM-free macro chunking wave on the Gemini baseline. Token 512/64 improved Precision@k slightly but lost Recall@k/NDCG@k and was slower; token 2048/256 and sentence 1024/128 matched quality but added latency, so token 1024/128 remains the current retrieval baseline. Details: `studies/rag-foundation/README.md`.

## Current retrieval baseline

`gemini-embedding-001` + full SciFact + `prep-basic-clean-v1` + token chunks `1024/128` + Qdrant HNSW cosine + dense top-k5 + no reranker + no tool policy + no LLM generation.

## Next LLM-free macro cases

Keep the Gemini + token 1024/128 baseline frozen and compare search strategy:

1. dense top-k5 control.
2. dense top-k10 recall variant.
3. dense MMR diversity variant.
4. hybrid RRF lexical+dense variant.

## Quickstart

```bash
cp .env.example .env
uv sync --dev
docker compose up -d
just resolve
```
