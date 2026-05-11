set shell := ["zsh", "-cu"]

study := "studies/rag-foundation"
question := "What tool manages Python dependencies?"

setup:
    uv sync --dev

datasets:
    uv run mirage datasets

up:
    docker compose up -d

down:
    docker compose down

resolve:
    uv run mirage resolve --experiment {{study}}

ingest:
    uv run mirage ingest --experiment {{study}} --reset

ask:
    uv run mirage ask --experiment {{study}} --set generation_model_id='"gen-llama-3.1-8b"' "{{question}}"

eval:
    uv run mirage eval --experiment {{study}}

eval-retrieval exp="baseline":
    uv run mirage eval-retrieval --experiment {{study}} --set study_experiment_id='"{{exp}}"'

test:
    uv run pytest

lint:
    uv run ruff check .
