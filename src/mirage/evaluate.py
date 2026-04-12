from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from mirage.config import AppConfig
from mirage.io_utils import read_jsonl, write_jsonl
from mirage.metrics import (
    citation_hit_rate,
    exact_match,
    hit_at_k,
    mean_metric,
    percentile,
    reciprocal_rank,
    recall_at_k,
    token_f1,
)
from mirage.pipeline import answer_question
from mirage.schemas import EvalExample, RunMetrics


def evaluate_dataset(config: AppConfig, input_path: str) -> dict[str, str | float | int]:
    examples = read_jsonl(input_path, EvalExample)
    if not examples:
        raise ValueError(f"No evaluation examples found in {input_path}")

    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{config.baseline.name}"
    run_dir = Path(config.evaluation.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    hit_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    exact_match_scores: list[float] = []
    token_f1_scores: list[float] = []
    citation_scores: list[float] = []
    latencies_ms: list[float] = []
    costs_usd: list[float] = []

    for example in examples:
        answer, retrieved_items = answer_question(config, example.question, qid=example.qid)

        hit_score = hit_at_k(answer.retrieved_doc_ids, example.gold_doc_ids, config.retrieval.top_k)
        recall_score = recall_at_k(answer.retrieved_doc_ids, example.gold_doc_ids, config.retrieval.top_k)
        mrr_score = reciprocal_rank(answer.retrieved_doc_ids, example.gold_doc_ids, config.retrieval.mrr_depth)
        em_score = exact_match(answer.answer, example.gold_answers)
        f1_score = token_f1(answer.answer, example.gold_answers)
        citation_score = citation_hit_rate(answer.citations, example.gold_doc_ids)

        hit_scores.append(hit_score)
        recall_scores.append(recall_score)
        mrr_scores.append(mrr_score)
        exact_match_scores.append(em_score)
        token_f1_scores.append(f1_score)
        citation_scores.append(citation_score)
        latencies_ms.append(answer.latency_ms)
        costs_usd.append(answer.estimated_cost_usd)

        retrieval_rows.append(
            {
                "qid": example.qid,
                "question": example.question,
                "gold_doc_ids": example.gold_doc_ids,
                "retrieved": [item.model_dump() for item in retrieved_items],
                "metrics": {
                    "hit_at_5": hit_score,
                    "recall_at_5": recall_score,
                    "mrr_at_10": mrr_score,
                },
            }
        )
        answer_rows.append(
            {
                "qid": example.qid,
                "question": example.question,
                "gold_answers": example.gold_answers,
                **answer.model_dump(),
                "metrics": {
                    "exact_match": em_score,
                    "token_f1": f1_score,
                    "citation_hit_rate": citation_score,
                },
            }
        )

    metrics = RunMetrics(
        retrieval={
            "hit_at_5": round(mean_metric(hit_scores), 4),
            "recall_at_5": round(mean_metric(recall_scores), 4),
            "mrr_at_10": round(mean_metric(mrr_scores), 4),
        },
        generation={
            "exact_match": round(mean_metric(exact_match_scores), 4),
            "token_f1": round(mean_metric(token_f1_scores), 4),
            "citation_hit_rate": round(mean_metric(citation_scores), 4),
        },
        operational={
            "examples": len(examples),
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 2),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 2),
            "estimated_cost_usd": round(sum(costs_usd), 6),
        },
    )

    write_jsonl(run_dir / "retrieval.jsonl", retrieval_rows)
    write_jsonl(run_dir / "answers.jsonl", answer_rows)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics.model_dump(), handle, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "examples": len(examples),
        "hit_at_5": metrics.retrieval["hit_at_5"],
        "exact_match": metrics.generation["exact_match"],
        "estimated_cost_usd": metrics.operational["estimated_cost_usd"],
    }
