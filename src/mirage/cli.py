from __future__ import annotations

import json
from pathlib import Path

import typer

from mirage.config import load_config
from mirage.evaluate import evaluate_dataset
from mirage.ingest import ingest_corpus
from mirage.pipeline import answer_question

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ingest(
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    reset: bool = typer.Option(False, "--reset", help="Recreate the collection before indexing."),
) -> None:
    settings = load_config(config_path)
    result = ingest_corpus(settings, str(input_path), reset=reset)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def ask(
    question: str = typer.Argument(...),
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    settings = load_config(config_path)
    answer, retrieved_items = answer_question(settings, question)
    typer.echo(answer.answer)
    typer.echo("")
    typer.echo("Retrieved context:")
    for item in retrieved_items:
        typer.echo(f"- {item.doc_id} :: {item.chunk_id} :: {item.score:.4f}")


@app.command()
def eval(
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    settings = load_config(config_path)
    result = evaluate_dataset(settings, str(input_path))
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
