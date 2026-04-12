# AGENTS

## Working conventions
- Manage local services with `docker compose` and override files for easy rebuilds.
- Manage Python dependencies with `uv`.
- Run Python entrypoints from the project environment via `uv run`.
- Use `just` as the main entrypoint for common workflows.
- Keep testing pragmatic rather than exhaustive for every experiment.

## Experiment governance
- Treat `EXPERIMENTS.md` as the canonical system contract for experiment axes, config layering, artifact reuse, naming, cost accounting, and report linkage.
- Treat `experiments/` as the only home for concrete experiment definitions.
- Treat `reports/SCHEMA.md` as the canonical contract for human-readable reports.
- Keep experiment implementation layered: source data -> adapter-normalized records -> prepared data -> chunks -> embeddings -> store artifacts -> retrieval artifacts -> answer artifacts -> reports.
- Fixed generation model set for the first large comparison group: `meta-llama/llama-3.1-8b-instruct`, `x-ai/grok-4-fast`, `minimax/minimax-m2.7`.
- Fixed initial embedding model set for the first large comparison group: local `BAAI/bge-small-en-v1.5`, `openai/text-embedding-3-small`, `openai/text-embedding-3-large`.
- Default executable baseline for the first large comparison group uses `openai/text-embedding-3-small` for both store-time and query-time embeddings through OpenRouter.
- Price all external LLM and embedding usage against the pinned OpenRouter catalog defined in `EXPERIMENTS.md`.
- Semantic chunking is model-bound: chunking artifacts produced by one semantic chunker model are not interchangeable with artifacts produced by another semantic chunker model.
- Experiment directories must contain `README.md` and `overlay.toml`; `hooks.py` is optional and is used only when config composition alone is insufficient.
- Reports contain observations and conclusions only. They must not contain next steps, action plans, or roadmap language.

## Current repository rules
- Keep external benchmark source manifests under `data/datasets/<dataset_id>/` with a dataset `README.md` and machine-readable source metadata.
- Keep downloaded benchmark payloads out of git under dataset-local `raw/` or `downloads/` directories.
- Do not persist repository-local normalized benchmark copies for external datasets; adapt upstream payloads into repository `Document` and `EvalExample` records at runtime before ingest or evaluation.
- Keep smoke-test corpus data in JSONL under `data/corpus/`.
- Keep smoke-test evaluation data in JSONL under `data/eval/` with `gold_answers` and doc-level `gold_doc_ids`.
- Register benchmark dataset adapters and roots in `configs/registries/data.toml`.
- Save machine-readable experiment outputs under `runs/` using the layered artifact layout defined in `EXPERIMENTS.md`.
- Save human-readable reports under `reports/` using `reports/SCHEMA.md`.
