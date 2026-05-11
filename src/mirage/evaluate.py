from __future__ import annotations

import json

from mirage.adapters import load_eval_examples_for_spec
from mirage.artifacts import ArtifactLayout
from mirage.config import ResolvedSpec
from mirage.io_utils import write_json, write_jsonl
from mirage.metrics import (
    citation_hit_rate,
    exact_match,
    hit_at_k,
    mean_metric,
    ndcg_at_k,
    percentile,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    reranker_coverage,
    token_f1,
)
from mirage.pipeline import _apply_tool_policy, answer_question
from mirage.schemas import RunMetrics


def evaluate_retrieval_spec(spec: ResolvedSpec, reset: bool = False) -> dict[str, object]:
    examples = load_eval_examples_for_spec(spec)
    if not examples:
        raise ValueError(
            f"No evaluation examples found for eval adapter '{spec.eval_adapter_id}' and split '{spec.eval_split}'"
        )

    layout = ArtifactLayout(spec)
    retrieval_dir = layout.retrieval_dir()
    if reset and retrieval_dir.exists():
        for path in retrieval_dir.iterdir():
            path.unlink() if path.is_file() else None

    retrieval_rows: list[dict] = []
    hit_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []
    latencies_ms: list[float] = []
    query_embedding_costs: list[float] = []
    reranker_coverages: list[float] = []

    for example in examples:
        retrieval, retrieved_items = _apply_tool_policy(spec, example.question, qid=example.qid)
        hit_score = hit_at_k(retrieval.retrieved_doc_ids, example.gold_doc_ids, spec.top_k)
        precision_score = precision_at_k(retrieval.retrieved_doc_ids, example.gold_doc_ids, spec.top_k)
        recall_score = recall_at_k(retrieval.retrieved_doc_ids, example.gold_doc_ids, spec.top_k)
        mrr_score = reciprocal_rank(retrieval.retrieved_doc_ids, example.gold_doc_ids, spec.mrr_depth)
        ndcg_score = ndcg_at_k(retrieval.retrieved_doc_ids, example.gold_doc_ids, spec.mrr_depth)

        hit_scores.append(hit_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        mrr_scores.append(mrr_score)
        ndcg_scores.append(ndcg_score)
        latencies_ms.append(retrieval.retrieval_latency_ms)
        query_embedding_costs.append(retrieval.query_embedding_cost_usd)
        reranker_coverages.append(reranker_coverage(spec.search_dense_top_k or spec.top_k, spec.top_k))
        retrieval_rows.append(
            {
                "qid": example.qid,
                "question": example.question,
                "gold_doc_ids": example.gold_doc_ids,
                "retrieval": retrieval.model_dump(mode="json"),
                "retrieved": [item.model_dump(mode="json") for item in retrieved_items],
                "metrics": {
                    "hit_at_k": hit_score,
                    "precision_at_k": precision_score,
                    "recall_at_k": recall_score,
                    "mrr_at_k": mrr_score,
                    "ndcg_at_k": ndcg_score,
                },
            }
        )

    metrics = RunMetrics(
        retrieval={
            "hit_at_k": round(mean_metric(hit_scores), 4),
            "precision_at_k": round(mean_metric(precision_scores), 4),
            "recall_at_k": round(mean_metric(recall_scores), 4),
            "mrr_at_k": round(mean_metric(mrr_scores), 4),
            "ndcg_at_k": round(mean_metric(ndcg_scores), 4),
        },
        generation={
            "exact_match": 0.0,
            "token_f1": 0.0,
            "citation_hit_rate": 0.0,
        },
        operational={
            "examples": float(len(examples)),
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 2),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 2),
            "reranker_coverage": round(mean_metric(reranker_coverages), 4),
        },
        cost={
            "build_preprocessing_cost_usd": 0.0,
            "build_chunking_cost_usd": 0.0,
            "build_store_embedding_cost_usd": 0.0,
            "build_total_external_cost_usd": 0.0,
            "mean_query_embedding_cost_usd_eval": round(mean_metric(query_embedding_costs), 6),
            "mean_generation_input_cost_usd_eval": 0.0,
            "mean_generation_output_cost_usd_eval": 0.0,
            "mean_query_cost_usd_eval": round(mean_metric(query_embedding_costs), 6),
            "projected_query_cost_1m_usd": round(mean_metric(query_embedding_costs) * 1_000_000, 2),
            "one_time_build_cost_usd": 0.0,
            "amortized_total_cost_1m_usd": round(mean_metric(query_embedding_costs) * 1_000_000, 2),
        },
    )
    write_json(retrieval_dir / "resolved_spec.json", spec.model_dump(mode="json", exclude={"env"}))
    write_jsonl(retrieval_dir / "retrieval.jsonl", retrieval_rows)
    write_json(retrieval_dir / "retrieval_metrics.json", metrics.model_dump(mode="json"))
    return {
        "group_id": spec.group_id,
        "experiment_id": spec.experiment_id,
        "dataset_id": spec.dataset_id,
        "evalset_id": spec.evalset_id,
        "retrieval_variant_id": f"{spec.query_embedding_model_id}__{spec.search_algorithm_id}__{spec.reranker_id}__{spec.tool_policy_id}",
        "examples": len(examples),
        "retrieval_dir": str(retrieval_dir),
        "hit_at_k": metrics.retrieval["hit_at_k"],
        "precision_at_k": metrics.retrieval["precision_at_k"],
        "recall_at_k": metrics.retrieval["recall_at_k"],
        "mrr_at_k": metrics.retrieval["mrr_at_k"],
        "ndcg_at_k": metrics.retrieval["ndcg_at_k"],
        "latency_p50_ms": metrics.operational["latency_p50_ms"],
        "latency_p95_ms": metrics.operational["latency_p95_ms"],
        "mean_query_embedding_cost_usd_eval": metrics.cost["mean_query_embedding_cost_usd_eval"],
    }


def evaluate_spec(spec: ResolvedSpec, reset: bool = False) -> dict[str, object]:
    examples = load_eval_examples_for_spec(spec)
    if not examples:
        raise ValueError(
            f"No evaluation examples found for eval adapter '{spec.eval_adapter_id}' and split '{spec.eval_split}'"
        )

    if reset:
        answers_dir = ArtifactLayout(spec).answers_dir()
        retrieval_dir = ArtifactLayout(spec).retrieval_dir()
        if answers_dir.exists():
            for path in answers_dir.iterdir():
                path.unlink() if path.is_file() else None
        if retrieval_dir.exists():
            for path in retrieval_dir.iterdir():
                path.unlink() if path.is_file() else None

    layout = ArtifactLayout(spec)
    retrieval_dir = layout.retrieval_dir()
    answers_dir = layout.answers_dir()

    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    hit_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []
    exact_match_scores: list[float] = []
    token_f1_scores: list[float] = []
    citation_scores: list[float] = []
    latencies_ms: list[float] = []
    query_embedding_costs: list[float] = []
    generation_input_costs: list[float] = []
    generation_output_costs: list[float] = []
    reranker_coverages: list[float] = []

    for example in examples:
        retrieval, answer, retrieved_items = answer_question(spec, example.question, qid=example.qid)

        hit_score = hit_at_k(answer.retrieved_doc_ids, example.gold_doc_ids, spec.top_k)
        precision_score = precision_at_k(answer.retrieved_doc_ids, example.gold_doc_ids, spec.top_k)
        recall_score = recall_at_k(answer.retrieved_doc_ids, example.gold_doc_ids, spec.top_k)
        mrr_score = reciprocal_rank(answer.retrieved_doc_ids, example.gold_doc_ids, spec.mrr_depth)
        ndcg_score = ndcg_at_k(answer.retrieved_doc_ids, example.gold_doc_ids, spec.mrr_depth)
        em_score = exact_match(answer.answer, example.gold_answers)
        f1_score = token_f1(answer.answer, example.gold_answers)
        citation_score = citation_hit_rate(answer.citations, example.gold_doc_ids)

        hit_scores.append(hit_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        mrr_scores.append(mrr_score)
        ndcg_scores.append(ndcg_score)
        exact_match_scores.append(em_score)
        token_f1_scores.append(f1_score)
        citation_scores.append(citation_score)
        latencies_ms.append(answer.latency_ms)
        query_embedding_costs.append(retrieval.query_embedding_cost_usd)
        generation_input_costs.append(answer.generation_input_cost_usd)
        generation_output_costs.append(answer.generation_output_cost_usd)
        reranker_coverages.append(reranker_coverage(spec.search_dense_top_k or spec.top_k, spec.top_k))

        retrieval_rows.append(
            {
                "qid": example.qid,
                "question": example.question,
                "gold_doc_ids": example.gold_doc_ids,
                "retrieval": retrieval.model_dump(mode="json"),
                "retrieved": [item.model_dump(mode="json") for item in retrieved_items],
                "metrics": {
                    "hit_at_k": hit_score,
                    "precision_at_k": precision_score,
                    "recall_at_k": recall_score,
                    "mrr_at_k": mrr_score,
                    "ndcg_at_k": ndcg_score,
                },
            }
        )
        answer_rows.append(
            {
                "qid": example.qid,
                "question": example.question,
                "gold_answers": example.gold_answers,
                "answer": answer.model_dump(mode="json"),
                "metrics": {
                    "exact_match": em_score,
                    "token_f1": f1_score,
                    "citation_hit_rate": citation_score,
                },
            }
        )

    build_store_metadata_path = layout.store_dir() / "metadata.json"
    if build_store_metadata_path.exists():
        build_store_metadata = json.loads(build_store_metadata_path.read_text(encoding="utf-8"))
    else:
        build_store_metadata = {}

    mean_query_cost = mean_metric(
        [q + gi + go for q, gi, go in zip(query_embedding_costs, generation_input_costs, generation_output_costs, strict=True)]
    )
    projected_query_cost_1m_usd = round(mean_query_cost * 1_000_000, 2)
    one_time_build_cost_usd = round(float(build_store_metadata.get("build_store_embedding_cost_usd", 0.0)), 6)

    metrics = RunMetrics(
        retrieval={
            "hit_at_k": round(mean_metric(hit_scores), 4),
            "precision_at_k": round(mean_metric(precision_scores), 4),
            "recall_at_k": round(mean_metric(recall_scores), 4),
            "mrr_at_k": round(mean_metric(mrr_scores), 4),
            "ndcg_at_k": round(mean_metric(ndcg_scores), 4),
        },
        generation={
            "exact_match": round(mean_metric(exact_match_scores), 4),
            "token_f1": round(mean_metric(token_f1_scores), 4),
            "citation_hit_rate": round(mean_metric(citation_scores), 4),
        },
        operational={
            "examples": float(len(examples)),
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 2),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 2),
            "reranker_coverage": round(mean_metric(reranker_coverages), 4),
        },
        cost={
            "build_preprocessing_cost_usd": 0.0,
            "build_chunking_cost_usd": 0.0,
            "build_store_embedding_cost_usd": round(float(build_store_metadata.get("build_store_embedding_cost_usd", 0.0)), 6),
            "build_total_external_cost_usd": one_time_build_cost_usd,
            "mean_query_embedding_cost_usd_eval": round(mean_metric(query_embedding_costs), 6),
            "mean_generation_input_cost_usd_eval": round(mean_metric(generation_input_costs), 6),
            "mean_generation_output_cost_usd_eval": round(mean_metric(generation_output_costs), 6),
            "mean_query_cost_usd_eval": round(mean_query_cost, 6),
            "projected_query_cost_1m_usd": projected_query_cost_1m_usd,
            "one_time_build_cost_usd": one_time_build_cost_usd,
            "amortized_total_cost_1m_usd": round(one_time_build_cost_usd + projected_query_cost_1m_usd, 2),
        },
    )

    write_json(retrieval_dir / "resolved_spec.json", spec.model_dump(mode="json", exclude={"env"}))
    write_json(answers_dir / "resolved_spec.json", spec.model_dump(mode="json", exclude={"env"}))
    write_jsonl(retrieval_dir / "retrieval.jsonl", retrieval_rows)
    write_jsonl(answers_dir / "answers.jsonl", answer_rows)
    write_json(answers_dir / "metrics.json", metrics.model_dump(mode="json"))

    return {
        "group_id": spec.group_id,
        "experiment_id": spec.experiment_id,
        "generation_model_id": spec.generation_model_id,
        "examples": len(examples),
        "retrieval_dir": str(retrieval_dir),
        "answers_dir": str(answers_dir),
        "hit_at_k": metrics.retrieval["hit_at_k"],
        "exact_match": metrics.generation["exact_match"],
        "mean_query_cost_usd_eval": metrics.cost["mean_query_cost_usd_eval"],
    }
