from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from mirage.datasets import fetch_all_datasets, fetch_dataset
from mirage.pipeline import answer_question
from mirage.reporting import synthesize_reports
from mirage.runner import describe_spec, persist_resolved_specs, resolve_specs, run_eval, run_ingest, select_single_spec

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
def datasets(
    dataset_id: Optional[str] = typer.Option(None, "--dataset-id"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    if dataset_id:
        result = fetch_dataset(dataset_id, force=force)
    else:
        result = fetch_all_datasets(force=force)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def eval(
    experiment: Path = typer.Option(..., "--experiment", exists=True, file_okay=False, dir_okay=True),
    set_values: Optional[List[str]] = typer.Option(None, "--set"),
    reset: bool = typer.Option(False, "--reset", help="Recreate matching store artifacts before evaluation."),
    report: bool = typer.Option(False, "--report", help="Synthesize markdown report after evaluation."),
    baseline_id: Optional[str] = typer.Option(None, "--baseline-id"),
) -> None:
    result = run_eval(
        experiment,
        overrides=set_values,
        reset=reset,
        synthesize_report=report,
        baseline_id=baseline_id,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def report(
    experiment: Path = typer.Option(..., "--experiment", exists=True, file_okay=False, dir_okay=True),
    set_values: Optional[List[str]] = typer.Option(None, "--set"),
    baseline_id: Optional[str] = typer.Option(None, "--baseline-id"),
) -> None:
    report_paths = [
        str(path) for path in synthesize_reports(resolve_specs(experiment, set_values), baseline_id=baseline_id)
    ]
    typer.echo(json.dumps({"report_paths": report_paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
