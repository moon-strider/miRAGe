# miRAGe

Minimal platform for controlled RAG experiments.

## Layout

- `config/` — global defaults and registries.
- `datasets/` — smoke fixtures and external benchmark source folders.
- `studies/` — executable study definitions and human results.
- `artifacts/` — ignored machine outputs and caches.
- `src/mirage/` — runtime code.
- `tests/` — regression tests.

## Quickstart

```bash
cp .env.example .env
uv sync --dev
docker compose up -d
just resolve
```
