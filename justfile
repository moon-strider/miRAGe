set shell := ["zsh", "-cu"]

experiment := "experiments/01-rag-foundation/baseline-freeze"
question := "What tool manages Python dependencies?"
chunking_experiment := "experiments/01-rag-foundation/load-deterministic-chunking"

setup:
    uv sync --dev

datasets:
    uv run mirage datasets

up:
    docker compose up -d

down:
    docker compose down

resolve:
    uv run mirage resolve --experiment {{experiment}}

resolve-chunking:
    uv run mirage resolve --experiment {{chunking_experiment}} --set generation_model_id='"gen-llama-3.1-8b"'

ingest:
    uv run mirage ingest --experiment {{experiment}} --reset

ingest-chunking:
    uv run mirage ingest --experiment {{chunking_experiment}} --reset

ask:
    uv run mirage ask --experiment {{experiment}} --set generation_model_id='"gen-llama-3.1-8b"' "{{question}}"

eval:
    uv run mirage eval --experiment {{experiment}}

lint:
    uv run ruff check .

test:
    uv run pytest
