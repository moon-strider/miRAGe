from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from mirage.pipeline import answer_question
from mirage.runner import describe_spec, persist_resolved_specs, run_eval, run_ingest, select_single_spec

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def resolve(
    experiment: Path = typer.Option(..., "--experiment", exists=True, file_okay=False, dir_okay=True),
    output: Optional[Path] = typer.Option(None, "--output", file_okay=True, dir_okay=False),
    set_values: Optional[List[str]] = typer.Option(None, "--set"),
) -> None:
    result = persist_resolved_specs(experiment, output_path=output, overrides=set_values)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def ingest(
    experiment: Path = typer.Option(..., "--experiment", exists=True, file_okay=False, dir_okay=True),
    set_values: Optional[List[str]] = typer.Option(None, "--set"),
    reset: bool = typer.Option(False, "--reset", help="Recreate matching store artifacts before indexing."),
) -> None:
    result = run_ingest(experiment, overrides=set_values, reset=reset)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def ask(
    question: str = typer.Argument(...),
    experiment: Path = typer.Option(..., "--experiment", exists=True, file_okay=False, dir_okay=True),
    set_values: Optional[List[str]] = typer.Option(None, "--set"),
) -> None:
    spec = select_single_spec(experiment, set_values)
    retrieval, answer, retrieved_items = answer_question(spec, question)
    typer.echo(answer.answer)
    typer.echo("")
    typer.echo("Resolved spec:")
    typer.echo(json.dumps(describe_spec(spec), ensure_ascii=False, indent=2))
    typer.echo("")
    typer.echo("Retrieval:")
    typer.echo(json.dumps(retrieval.model_dump(mode="json"), ensure_ascii=False, indent=2))
    typer.echo("")
    typer.echo("Retrieved context:")
    for item in retrieved_items:
        typer.echo(f"- {item.doc_id} :: {item.chunk_id} :: {item.score:.4f}")


@app.command()
def eval(
    experiment: Path = typer.Option(..., "--experiment", exists=True, file_okay=False, dir_okay=True),
    set_values: Optional[List[str]] = typer.Option(None, "--set"),
    reset: bool = typer.Option(False, "--reset", help="Recreate matching store artifacts before evaluation."),
) -> None:
    result = run_eval(experiment, overrides=set_values, reset=reset)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
