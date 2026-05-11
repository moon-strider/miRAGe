from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mirage.artifacts import ArtifactLayout
from mirage.config import ResolvedSpec
from mirage.io_utils import read_json, read_jsonl
from mirage.registry import PROJECT_ROOT

_PRIMARY_AXIS_BY_EXPERIMENT = {
    "load-deterministic-chunking": "chunking_variant_id",
    "load-semantic-chunking": "chunking_model_id",
    "store-embedding-models": "store_embedding_model_id",
    "store-index-variants": "store_index_variant_id",
    "store-backends": "store_backend_id",
    "inference-search": "search_algorithm_id",
    "inference-prompting": "prompt_variant_id",
    "inference-agentic": "tool_policy_id",
}

_DEFAULT_BASELINE_BY_EXPERIMENT = {
    "load-deterministic-chunking": "chunk-token-1024-128-v1",
    "load-semantic-chunking": None,
    "store-embedding-models": "emb-bge-small-en-v1.5",
    "store-index-variants": "idx-qdrant-hnsw-cosine-default-v1",
    "store-backends": "qdrant",
    "inference-search": "search-dense-topk5-v1",
    "inference-prompting": "prompt-grounded-citations-v1",
    "inference-agentic": "none",
}

_AXIS_LABELS = {
    "preprocessing_variant_id": "preprocessing",
    "chunking_variant_id": "chunking",
    "chunking_model_id": "chunking model",
    "store_backend_id": "store backend",
    "store_index_variant_id": "index variant",
    "store_embedding_model_id": "store embedding",
    "query_embedding_model_id": "query embedding",
    "search_algorithm_id": "search algorithm",
    "reranker_id": "reranker",
    "prompt_variant_id": "prompt variant",
    "tool_policy_id": "tool policy",
}


def _slug(value: str) -> str:
    return value.replace("/", "-")


def _load_answers(spec: ResolvedSpec) -> list[dict[str, Any]]:
    return read_jsonl(ArtifactLayout(spec).answers_dir() / "answers.jsonl", dict)


def _load_retrieval(spec: ResolvedSpec) -> list[dict[str, Any]]:
    return read_jsonl(ArtifactLayout(spec).retrieval_dir() / "retrieval.jsonl", dict)


def _load_metrics(spec: ResolvedSpec) -> dict[str, Any]:
    return read_json(ArtifactLayout(spec).answers_dir() / "metrics.json")


def _group_specs_by_axis(specs: list[ResolvedSpec], axis_field: str) -> dict[str, list[ResolvedSpec]]:
    grouped: dict[str, list[ResolvedSpec]] = defaultdict(list)
    for spec in specs:
        grouped[str(getattr(spec, axis_field))].append(spec)
    return {key: sorted(value, key=lambda item: item.generation_model_id) for key, value in grouped.items()}


def _artifact_rows(baseline: ResolvedSpec, candidate: ResolvedSpec | None, experiment_id: str) -> list[tuple[str, str, str, str]]:
    axis_field = _PRIMARY_AXIS_BY_EXPERIMENT.get(experiment_id)
    if experiment_id == "baseline-freeze" or candidate is None or axis_field is None:
        layout = ArtifactLayout(baseline)
        return [
            ("prepared", str(layout.prepared_dir()), "rebuilt", "baseline freeze materialized this layer"),
            ("chunks", str(layout.chunks_dir()), "rebuilt", "baseline freeze materialized this layer"),
            ("store", str(layout.store_dir()), "rebuilt", "baseline freeze materialized this layer"),
            ("retrieval", str(layout.retrieval_dir()), "rebuilt", "baseline freeze materialized this layer"),
            ("answers", str(layout.answers_dir()), "rebuilt", "baseline freeze materialized this layer"),
        ]
    baseline_layout = ArtifactLayout(baseline)
    candidate_layout = ArtifactLayout(candidate)
    if axis_field in {"chunking_variant_id", "chunking_model_id"}:
        statuses = {
            "prepared": ("reused", "chunking change keeps prepared documents reusable"),
            "chunks": ("rebuilt", "chunking change invalidates chunk artifacts"),
            "store": ("rebuilt", "store depends on chunk artifacts"),
            "retrieval": ("rebuilt", "retrieval depends on rebuilt store artifacts"),
            "answers": ("rebuilt", "answers depend on rebuilt retrieval artifacts"),
        }
    elif axis_field in {"store_backend_id", "store_index_variant_id", "store_embedding_model_id"}:
        statuses = {
            "prepared": ("reused", "store-time change keeps prepared documents reusable"),
            "chunks": ("reused", "store-time change keeps chunk artifacts reusable"),
            "store": ("rebuilt", "store-time change invalidates store artifacts"),
            "retrieval": ("rebuilt", "retrieval depends on rebuilt store artifacts"),
            "answers": ("rebuilt", "answers depend on rebuilt retrieval artifacts"),
        }
    elif axis_field in {"search_algorithm_id", "reranker_id"}:
        statuses = {
            "prepared": ("reused", "search-time change keeps upstream artifacts reusable"),
            "chunks": ("reused", "search-time change keeps upstream artifacts reusable"),
            "store": ("reused", "search-time change keeps store artifacts reusable"),
            "retrieval": ("rebuilt", "retrieval behavior changed"),
            "answers": ("rebuilt", "answers depend on retrieval output"),
        }
    elif axis_field in {"prompt_variant_id", "tool_policy_id"}:
        statuses = {
            "prepared": ("reused", "inference-time change keeps upstream artifacts reusable"),
            "chunks": ("reused", "inference-time change keeps upstream artifacts reusable"),
            "store": ("reused", "inference-time change keeps store artifacts reusable"),
            "retrieval": ("reused", "retrieval stack is unchanged"),
            "answers": ("rebuilt", "answer generation behavior changed"),
        }
    else:
        statuses = {
            "prepared": ("reused", "upstream layer unchanged"),
            "chunks": ("reused", "upstream layer unchanged"),
            "store": ("reused", "upstream layer unchanged"),
            "retrieval": ("rebuilt", "comparison candidate has a distinct eval run"),
            "answers": ("rebuilt", "comparison candidate has a distinct eval run"),
        }
    return [
        ("prepared", f"baseline={baseline_layout.prepared_dir()} candidate={candidate_layout.prepared_dir()}", *statuses["prepared"]),
        ("chunks", f"baseline={baseline_layout.chunks_dir()} candidate={candidate_layout.chunks_dir()}", *statuses["chunks"]),
        ("store", f"baseline={baseline_layout.store_dir()} candidate={candidate_layout.store_dir()}", *statuses["store"]),
        ("retrieval", f"baseline={baseline_layout.retrieval_dir()} candidate={candidate_layout.retrieval_dir()}", *statuses["retrieval"]),
        ("answers", f"baseline={baseline_layout.answers_dir()} candidate={candidate_layout.answers_dir()}", *statuses["answers"]),
    ]


def _compared_variants_table(baseline: ResolvedSpec, candidate: ResolvedSpec | None) -> str:
    rows = ["| Axis | Baseline | Candidate | Status |", "| --- | --- | --- | --- |"]
    baseline_generation_models = baseline.generation_model_id if candidate is None else "see metrics table"
    candidate_generation_models = baseline.generation_model_id if candidate is None else "see metrics table"
    for field, label in _AXIS_LABELS.items():
        baseline_value = getattr(baseline, field)
        candidate_value = getattr(candidate, field) if candidate is not None else baseline_value
        status = "changed" if baseline_value != candidate_value else "fixed"
        rows.append(f"| {label} | {baseline_value} | {candidate_value} | {status} |")
    rows.append(
        f"| generation models | {baseline_generation_models} | {candidate_generation_models} | {'fixed' if candidate is None else 'changed'} |"
    )
    return "\n".join(rows)


def _metrics_table(baseline_specs: list[ResolvedSpec], candidate_specs: list[ResolvedSpec] | None) -> str:
    rows = [
        "| generation_model_id | baseline Hit@k | candidate Hit@k | baseline Precision@k | candidate Precision@k | baseline Recall@k | candidate Recall@k | baseline MRR@k | candidate MRR@k | baseline NDCG@k | candidate NDCG@k | baseline Exact Match | candidate Exact Match | baseline Token F1 | candidate Token F1 | baseline Citation Hit Rate | candidate Citation Hit Rate | baseline p50 ms | candidate p50 ms | baseline p95 ms | candidate p95 ms | baseline reranker coverage | candidate reranker coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_by_model = {spec.generation_model_id: _load_metrics(spec) for spec in baseline_specs}
    candidate_by_model = baseline_by_model if candidate_specs is None else {spec.generation_model_id: _load_metrics(spec) for spec in candidate_specs}
    for generation_model_id in sorted(baseline_by_model):
        baseline_metrics = baseline_by_model[generation_model_id]
        candidate_metrics = candidate_by_model[generation_model_id]
        rows.append(
            "| {model} | {b_hit:.4f} | {c_hit:.4f} | {b_precision:.4f} | {c_precision:.4f} | {b_recall:.4f} | {c_recall:.4f} | {b_mrr:.4f} | {c_mrr:.4f} | {b_ndcg:.4f} | {c_ndcg:.4f} | {b_em:.4f} | {c_em:.4f} | {b_f1:.4f} | {c_f1:.4f} | {b_cit:.4f} | {c_cit:.4f} | {b_p50:.2f} | {c_p50:.2f} | {b_p95:.2f} | {c_p95:.2f} | {b_cov:.4f} | {c_cov:.4f} |".format(
                model=generation_model_id,
                b_hit=baseline_metrics["retrieval"]["hit_at_k"],
                c_hit=candidate_metrics["retrieval"]["hit_at_k"],
                b_precision=baseline_metrics["retrieval"]["precision_at_k"],
                c_precision=candidate_metrics["retrieval"]["precision_at_k"],
                b_recall=baseline_metrics["retrieval"]["recall_at_k"],
                c_recall=candidate_metrics["retrieval"]["recall_at_k"],
                b_mrr=baseline_metrics["retrieval"]["mrr_at_k"],
                c_mrr=candidate_metrics["retrieval"]["mrr_at_k"],
                b_ndcg=baseline_metrics["retrieval"]["ndcg_at_k"],
                c_ndcg=candidate_metrics["retrieval"]["ndcg_at_k"],
                b_em=baseline_metrics["generation"]["exact_match"],
                c_em=candidate_metrics["generation"]["exact_match"],
                b_f1=baseline_metrics["generation"]["token_f1"],
                c_f1=candidate_metrics["generation"]["token_f1"],
                b_cit=baseline_metrics["generation"]["citation_hit_rate"],
                c_cit=candidate_metrics["generation"]["citation_hit_rate"],
                b_p50=baseline_metrics["operational"]["latency_p50_ms"],
                c_p50=candidate_metrics["operational"]["latency_p50_ms"],
                b_p95=baseline_metrics["operational"]["latency_p95_ms"],
                c_p95=candidate_metrics["operational"]["latency_p95_ms"],
                b_cov=baseline_metrics["operational"]["reranker_coverage"],
                c_cov=candidate_metrics["operational"]["reranker_coverage"],
            )
        )
    return "\n".join(rows)


def _cost_table(baseline_specs: list[ResolvedSpec], candidate_specs: list[ResolvedSpec] | None) -> str:
    rows = [
        "| generation_model_id | baseline build preprocessing | candidate build preprocessing | baseline build chunking | candidate build chunking | baseline build store embeddings | candidate build store embeddings | baseline mean query cost over eval | candidate mean query cost over eval | baseline projected query cost at 1M | candidate projected query cost at 1M | baseline amortized total at 1M | candidate amortized total at 1M |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_by_model = {spec.generation_model_id: _load_metrics(spec) for spec in baseline_specs}
    candidate_by_model = baseline_by_model if candidate_specs is None else {spec.generation_model_id: _load_metrics(spec) for spec in candidate_specs}
    for generation_model_id in sorted(baseline_by_model):
        baseline_metrics = baseline_by_model[generation_model_id]["cost"]
        candidate_metrics = candidate_by_model[generation_model_id]["cost"]
        rows.append(
            "| {model} | {b_prep:.6f} | {c_prep:.6f} | {b_chunk:.6f} | {c_chunk:.6f} | {b_store:.6f} | {c_store:.6f} | {b_query:.6f} | {c_query:.6f} | {b_proj:.2f} | {c_proj:.2f} | {b_total:.2f} | {c_total:.2f} |".format(
                model=generation_model_id,
                b_prep=baseline_metrics["build_preprocessing_cost_usd"],
                c_prep=candidate_metrics["build_preprocessing_cost_usd"],
                b_chunk=baseline_metrics["build_chunking_cost_usd"],
                c_chunk=candidate_metrics["build_chunking_cost_usd"],
                b_store=baseline_metrics["build_store_embedding_cost_usd"],
                c_store=candidate_metrics["build_store_embedding_cost_usd"],
                b_query=baseline_metrics["mean_query_cost_usd_eval"],
                c_query=candidate_metrics["mean_query_cost_usd_eval"],
                b_proj=baseline_metrics["projected_query_cost_1m_usd"],
                c_proj=candidate_metrics["projected_query_cost_1m_usd"],
                b_total=baseline_metrics["amortized_total_cost_1m_usd"],
                c_total=candidate_metrics["amortized_total_cost_1m_usd"],
            )
        )
    return "\n".join(rows)


def _failure_cases(candidate_specs: list[ResolvedSpec]) -> str:
    rows: list[str] = []
    for spec in candidate_specs:
        answers = _load_answers(spec)
        retrieval = _load_retrieval(spec)
        retrieval_by_qid = {row["qid"]: row for row in retrieval}
        for row in answers[:2]:
            qid = row["qid"]
            retrieval_row = retrieval_by_qid.get(qid, {})
            answer_payload = row.get("answer", {})
            rows.append(
                "- qid: {qid}\n  question: {question}\n  expected answer summary: {expected}\n  retrieved document ids: {doc_ids}\n  model answer summary: {answer}\n  why the case matters: synthesized representative case from saved artifacts".format(
                    qid=qid,
                    question=row.get("question", ""),
                    expected=" | ".join(row.get("gold_answers", [])),
                    doc_ids=", ".join(retrieval_row.get("gold_doc_ids", []) or answer_payload.get("retrieved_doc_ids", [])),
                    answer=answer_payload.get("answer", ""),
                )
            )
        if rows:
            break
    return "\n".join(rows) if rows else "- no answer rows were available"


def _metadata_block(
    baseline_specs: list[ResolvedSpec],
    candidate_specs: list[ResolvedSpec] | None,
    baseline_id: str,
    candidate_id: str,
) -> str:
    spec = baseline_specs[0] if candidate_specs is None else candidate_specs[0]
    run_paths = [str(ArtifactLayout(item).answers_dir()) for item in (candidate_specs or baseline_specs)]
    payload = {
        "report_id": f"rpt-{spec.group_id}-{spec.experiment_id}-{_slug(candidate_id)}-vs-{_slug(baseline_id)}",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "group_id": spec.group_id,
        "experiment_id": spec.experiment_id,
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "dataset_id": spec.dataset_id,
        "evalset_id": spec.evalset_id,
        "preprocessing_variant_id": spec.preprocessing_variant_id,
        "chunking_variant_id": spec.chunking_variant_id,
        "chunking_model_id": spec.chunking_model_id,
        "load_variant_id": spec.load_variant_id,
        "store_backend_id": spec.store_backend_id,
        "store_index_variant_id": spec.store_index_variant_id,
        "store_embedding_model_id": spec.store_embedding_model_id,
        "query_embedding_model_id": spec.query_embedding_model_id,
        "search_algorithm_id": spec.search_algorithm_id,
        "reranker_id": spec.reranker_id,
        "prompt_variant_id": spec.prompt_variant_id,
        "tool_policy_id": spec.tool_policy_id,
        "inference_variant_id": spec.inference_variant_id,
        "generation_model_ids": [item.generation_model_id for item in (candidate_specs or baseline_specs)],
        "run_paths": run_paths,
        "price_catalog_version": spec.price_catalog_version,
    }
    yaml_lines = ["---"]
    for key, value in payload.items():
        if isinstance(value, list):
            yaml_lines.append(f"{key}:")
            for item in value:
                yaml_lines.append(f"  - {item}")
            continue
        yaml_lines.append(f"{key}: {value}")
    yaml_lines.append("---")
    return "\n".join(yaml_lines)


def _conclusion(baseline_specs: list[ResolvedSpec], candidate_specs: list[ResolvedSpec] | None, candidate_id: str) -> str:
    baseline_mean = sum(_load_metrics(spec)["generation"]["exact_match"] for spec in baseline_specs) / len(baseline_specs)
    candidate_mean = baseline_mean if candidate_specs is None else sum(
        _load_metrics(spec)["generation"]["exact_match"] for spec in candidate_specs
    ) / len(candidate_specs)
    if candidate_specs is None:
        return "This report freezes the baseline and records its measured metrics across the pinned generation models."
    if candidate_mean > baseline_mean:
        return f"{candidate_id} improved exact-match performance over the named baseline on the saved runs, with the full tradeoff visible in the metrics and cost tables above."
    if candidate_mean < baseline_mean:
        return f"{candidate_id} underperformed the named baseline on exact-match performance in the saved runs, despite any tradeoffs visible in latency or cost."
    return f"{candidate_id} matched the baseline on mean exact-match performance in the saved runs, so the decision depends on latency and cost tradeoffs."


def _write_report(
    reports_root: Path,
    baseline_specs: list[ResolvedSpec],
    candidate_specs: list[ResolvedSpec] | None,
    baseline_id: str,
    candidate_id: str,
) -> Path:
    spec = baseline_specs[0] if candidate_specs is None else candidate_specs[0]
    date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    report_dir = reports_root / spec.group_id / spec.experiment_id
    report_path = report_dir / f"{date_prefix}__{_slug(candidate_id)}__vs-{_slug(baseline_id)}.md"
    candidate_spec = None if candidate_specs is None else candidate_specs[0]
    artifact_rows = _artifact_rows(baseline_specs[0], candidate_spec, spec.experiment_id)
    artifact_table = ["| Layer | Artifact id or path | Reused or rebuilt | Why |", "| --- | --- | --- | --- |"]
    for layer, path, state, why in artifact_rows:
        artifact_table.append(f"| {layer} | {path} | {state} | {why} |")
    artifact_table_text = "\n".join(artifact_table)
    content = "\n\n".join(
        [
            _metadata_block(baseline_specs, candidate_specs, baseline_id, candidate_id),
            "## Goal\n\nState the measured difference between the named baseline and candidate using the saved experiment artifacts.",
            f"## Compared variants\n\n{_compared_variants_table(baseline_specs[0], candidate_spec)}",
            f"## Artifact reuse\n\n{artifact_table_text}",
            f"## Metrics summary\n\n{_metrics_table(baseline_specs, candidate_specs)}",
            f"## Cost summary\n\n{_cost_table(baseline_specs, candidate_specs)}",
            f"## Failure cases\n\n{_failure_cases(candidate_specs or baseline_specs)}",
            f"## Conclusion\n\n{_conclusion(baseline_specs, candidate_specs, candidate_id)}",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content + "\n", encoding="utf-8")
    return report_path


def synthesize_reports(
    specs: list[ResolvedSpec],
    *,
    reports_root: Path | None = None,
    baseline_id: str | None = None,
) -> list[Path]:
    if not specs:
        return []
    root = reports_root or (PROJECT_ROOT / "reports")
    experiment_id = specs[0].experiment_id
    if experiment_id == "baseline-freeze":
        return [_write_report(root, sorted(specs, key=lambda item: item.generation_model_id), None, "baseline-freeze", "baseline-freeze")]
    axis_field = _PRIMARY_AXIS_BY_EXPERIMENT.get(experiment_id)
    if axis_field is None:
        return []
    grouped = _group_specs_by_axis(specs, axis_field)
    resolved_baseline_id = baseline_id if baseline_id is not None else _DEFAULT_BASELINE_BY_EXPERIMENT.get(experiment_id)
    if resolved_baseline_id is None or resolved_baseline_id not in grouped:
        return []
    baseline_specs = grouped[resolved_baseline_id]
    report_paths: list[Path] = []
    for candidate_id, candidate_specs in grouped.items():
        if candidate_id == resolved_baseline_id:
            continue
        report_paths.append(_write_report(root, baseline_specs, candidate_specs, resolved_baseline_id, candidate_id))
    return report_paths
