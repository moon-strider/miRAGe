set shell := ["zsh", "-cu"]

config := "configs/baseline.toml"
corpus := "data/corpus/documents.jsonl"
evalset := "data/eval/dev.jsonl"

setup:
    uv sync --dev

up:
    docker compose up -d

down:
    docker compose down

ingest:
    uv run mirage ingest --config {{config}} --input {{corpus}} --reset

ask question:
    uv run mirage ask --config {{config}} "{{question}}"

eval:
    uv run mirage eval --config {{config}} --input {{evalset}}

lint:
    uv run ruff check .

test:
    uv run pytest
