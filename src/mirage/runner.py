from __future__ import annotations

from pathlib import Path

from mirage.artifacts import ArtifactLayout
from mirage.config import ResolvedSpec, load_experiment_specs
from mirage.evaluate import evaluate_retrieval_spec, evaluate_spec
from mirage.ingest import ingest_spec, prepare_documents, preflight_llm_chunking
from mirage.io_utils import write_json
from mirage.registry import parse_cli_overrides
from mirage.reporting import synthesize_reports


def resolve_specs(experiment_path: str | Path, overrides: list[str] | None = None) -> list[ResolvedSpec]:
    parsed_overrides = parse_cli_overrides(overrides)
    return load_experiment_specs(experiment_path, overrides=parsed_overrides)


def run_ingest(
    experiment_path: str | Path,
    *,
    overrides: list[str] | None = None,
    reset: bool = False,
) -> dict[str, object]:
    specs = resolve_specs(experiment_path, overrides)
    seen_store_keys: set[tuple[str, str]] = set()
    artifacts: list[dict[str, object]] = []
    for spec in specs:
        store_key = (spec.load_variant_id, spec.store_variant_id)
        if store_key in seen_store_keys:
            continue
        seen_store_keys.add(store_key)
        artifacts.append(ingest_spec(spec, reset=reset))
    return {
        "experiment": str(experiment_path),
        "resolved_specs": len(specs),
        "built_store_variants": len(artifacts),
        "artifacts": artifacts,
    }


def run_chunking_preflight(
    experiment_path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    specs = resolve_specs(experiment_path, overrides)
    seen_loads: set[str] = set()
    results: list[dict[str, object]] = []
    for spec in specs:
        if spec.load_variant_id in seen_loads or spec.chunking_kind != "semantic":
            continue
        seen_loads.add(spec.load_variant_id)
        documents, prepared_info = prepare_documents(spec, reset=False)
        preflight = preflight_llm_chunking(spec, documents)
        results.append(
            {
                "experiment_id": spec.experiment_id,
                "load_variant_id": spec.load_variant_id,
                "chunking_model_id": spec.chunking_model_id,
                "chunking_model": spec.semantic_chunking_model,
                "prepared_dir": prepared_info["prepared_dir"],
                **preflight.model_dump(mode="json"),
            }
        )
    return {"experiment": str(experiment_path), "semantic_load_variants": len(results), "results": results}


def run_retrieval_eval(
    experiment_path: str | Path,
    *,
    overrides: list[str] | None = None,
    reset: bool = False,
) -> dict[str, object]:
    retrieval_overrides = list(overrides or [])
    if not any(item.split("=", 1)[0] == "generation_model_id" for item in retrieval_overrides):
        retrieval_overrides.append("generation_model_id=none")
    specs = resolve_specs(experiment_path, retrieval_overrides)
    results: list[dict[str, object]] = []
    seen_store_keys: set[tuple[str, str]] = set()
    seen_retrieval_keys: set[tuple[str, str, str, str, str]] = set()
    for spec in specs:
        store_key = (spec.load_variant_id, spec.store_variant_id)
        if store_key not in seen_store_keys:
            ingest_spec(spec, reset=reset)
            seen_store_keys.add(store_key)
        retrieval_key = (
            spec.load_variant_id,
            spec.store_variant_id,
            spec.query_embedding_model_id,
            spec.search_algorithm_id,
            f"{spec.reranker_id}__{spec.tool_policy_id}",
        )
        if retrieval_key in seen_retrieval_keys:
            continue
        seen_retrieval_keys.add(retrieval_key)
        results.append(evaluate_retrieval_spec(spec, reset=reset))
    return {
        "experiment": str(experiment_path),
        "resolved_specs": len(specs),
        "evaluated_retrieval_specs": len(results),
        "results": results,
    }


def run_eval(
    experiment_path: str | Path,
    *,
    overrides: list[str] | None = None,
    reset: bool = False,
    synthesize_report: bool = False,
    baseline_id: str | None = None,
) -> dict[str, object]:
    specs = resolve_specs(experiment_path, overrides)
    results: list[dict[str, object]] = []
    seen_store_keys: set[tuple[str, str]] = set()
    for spec in specs:
        store_key = (spec.load_variant_id, spec.store_variant_id)
        if store_key not in seen_store_keys:
            ingest_spec(spec, reset=reset)
            seen_store_keys.add(store_key)
        results.append(evaluate_spec(spec, reset=reset))
    report_paths: list[str] = []
    if synthesize_report:
        report_paths = [str(path) for path in synthesize_reports(specs, baseline_id=baseline_id)]
    return {
        "experiment": str(experiment_path),
        "resolved_specs": len(specs),
        "evaluated_specs": len(results),
        "results": results,
        "report_paths": report_paths,
    }


def persist_resolved_specs(
    experiment_path: str | Path,
    *,
    output_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    specs = resolve_specs(experiment_path, overrides)
    payload = [spec.model_dump(mode="json", exclude={"env"}) for spec in specs]
    if output_path is not None:
        write_json(output_path, {"specs": payload})
    return {"specs": payload}


def select_single_spec(experiment_path: str | Path, overrides: list[str] | None = None) -> ResolvedSpec:
    specs = resolve_specs(experiment_path, overrides)
    if len(specs) != 1:
        raise ValueError(
            "The selected experiment resolves to multiple specs. Use overrides to narrow it to one point for ask."
        )
    return specs[0]


def describe_spec(spec: ResolvedSpec) -> dict[str, str]:
    layout = ArtifactLayout(spec)
    return {
        "group_id": spec.group_id,
        "experiment_id": spec.experiment_id,
        "load_variant_id": spec.load_variant_id,
        "store_variant_id": spec.store_variant_id,
        "inference_variant_id": spec.inference_variant_id,
        "generation_model_id": spec.generation_model_id,
        "store_dir": str(layout.store_dir()),
        "answers_dir": str(layout.answers_dir()),
    }
