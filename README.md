# miRAGe

Platform for controlled RAG experiments.

The repository is designed for repeatable comparison of preprocessing, chunking,
embedding models, storage backends, index settings, search strategies,
prompting methods, tool-use policies, and generation models.

## Quickstart

```bash
cp .env.example .env
# fill OPENROUTER_API_KEY in .env before commands that use remote models
uv sync --dev
docker compose up -d
just ingest
just ask "What tool manages Python dependencies?"
```

## Common commands

```bash
just setup
just up
just resolve
just ingest
just ask "What tool manages Python dependencies?"
just eval
just test
just lint
```

## Repository layout

- `configs/` — system config, registries, and reusable phase prototypes
- `data/corpus/` — corpus documents in JSONL
- `data/eval/` — evaluation sets in JSONL
- `experiments/` — concrete experiment groups and experiment overlays
- `reports/` — human-readable reports
- `runs/` — machine-readable artifacts
- `src/mirage/` — runtime, orchestration, and evaluation code
- `tests/` — pragmatic checks for core logic

## Notes

- Local services are managed with `docker compose` and override files.
- Python dependencies are managed with `uv`.
- Python commands are run from the project environment via `uv run` or the provided `just` recipes.
- Local embedding models can run without API keys.
- Remote generation and remote embedding calls use OpenRouter and require `OPENROUTER_API_KEY`.
- Semantic chunking and non-baseline tool policies are scaffolded in the experiment system but are not implemented yet.
