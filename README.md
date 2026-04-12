# miRAGe

Platform for controlled RAG experiments.

The repository is intended for comparing retrieval, chunking, preprocessing, indexing, prompting, and orchestration choices under repeatable conditions.

## Quickstart

```bash
cp .env.example .env
# fill OPENROUTER_API_KEY in .env before ask/eval
uv sync --dev
docker compose up -d
just ingest
just ask "What tool manages Python dependencies?"
```

## Common commands

```bash
just setup
just up
just ingest
just ask "What tool manages Python dependencies?"
just eval
just test
just lint
```

## Repository layout

- `configs/` — pinned experiment and baseline configs
- `data/corpus/` — corpus documents in JSONL
- `data/eval/` — evaluation sets in JSONL
- `reports/` — human-readable reports and benchmark summaries
- `runs/` — machine-readable run artifacts
- `src/mirage/` — ingestion, retrieval, answering, and evaluation code
- `tests/` — pragmatic checks for core logic

## Notes

- Local services are managed with `docker compose` and override files.
- Python dependencies are managed with `uv`.
- Python commands are run from the project environment via `uv run` or the provided `just` recipes.
- Embeddings are computed locally through FastEmbed.
- Answer generation uses OpenRouter and requires `OPENROUTER_API_KEY` in `.env` for `ask` and `eval`.

