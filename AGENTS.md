# AGENTS

## Working conventions
- Manage local services with `docker compose` and override files for easy rebuilds.
- Manage Python dependencies with `uv`.
- Run Python entrypoints from the project environment via `uv run`.
- Use `just` as the main entrypoint for common workflows.
- Keep testing pragmatic rather than exhaustive for every experiment.

## Current repository rules
- Keep corpus data in JSONL under `data/corpus/`.
- Keep evaluation data in JSONL under `data/eval/` with `gold_answers` and doc-level `gold_doc_ids`.
- Save machine-readable experiment outputs under `runs/`.
- Save human-readable reports and benchmark summaries under `reports/`.
