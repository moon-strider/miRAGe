# miRAGe

Platform for controlled RAG experiments.

The repository is designed for repeatable comparison of preprocessing, chunking,
embedding models, storage backends, index settings, search strategies,
prompting methods, tool-use policies, generation models, and dataset regimes.

The baseline experiment path uses OpenRouter-hosted `openai/text-embedding-3-small`
for both store-time and query-time embeddings, and uses the three pinned generation
models defined in the experiment registries.

## Quickstart

```bash
cp .env.example .env
# fill OPENROUTER_API_KEY in .env before baseline ingest/eval/ask commands
uv sync --dev
docker compose up -d
just ingest
just ask "What tool manages Python dependencies?"
```

## Common commands

```bash
just setup
just up
just datasets
just resolve
just ingest
just ask "What tool manages Python dependencies?"
just eval
just test
just lint

# run the first executable chunking comparison group
just resolve experiment=experiments/01-rag-foundation/load-deterministic-chunking
just ingest experiment=experiments/01-rag-foundation/load-deterministic-chunking
```

## Repository layout

- `configs/` — system config, registries, and reusable phase prototypes
- `data/datasets/` — benchmark source folders with dataset READMEs, source manifests, downloaded upstream payloads, and unpacked raw files used by adapters
- `data/corpus/` — committed smoke-test corpus documents in JSONL
- `data/eval/` — committed smoke-test evaluation sets in JSONL
- `experiments/` — concrete experiment groups and experiment overlays
- `reports/` — human-readable reports
- `runs/` — machine-readable artifacts
- `src/mirage/` — runtime, orchestration, evaluation code, and dataset adapters
- `tests/` — pragmatic checks for core logic

## Notes

- Local services are managed with `docker compose` and override files.
- Python dependencies are managed with `uv`.
- Python commands are run from the project environment via `uv run` or the provided `just` recipes.
- The baseline experiment defaults to OpenRouter for both embeddings and generation, so `OPENROUTER_API_KEY` is required for `just ingest`, `just ask`, and `just eval` unless an experiment override switches to a local embedding model.
- External benchmark payloads are downloaded into dataset-local `downloads/` and unpacked into `raw/`; they are not tracked in git.
- Use `uv run mirage datasets` or `just datasets` to fetch all declared benchmark archives into the expected local dataset directories.
- External datasets are adapted into repository `Document` and `EvalExample` records at runtime before ingest or evaluation; repository-local normalized copies are not stored.
- Semantic chunking and non-baseline tool policies are scaffolded in the experiment system but are not implemented yet.
