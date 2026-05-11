from __future__ import annotations

from pathlib import Path

import pytest

from mirage.config import load_experiment_specs
from mirage.evaluate import evaluate_retrieval_spec
from mirage.runner import run_retrieval_eval
from mirage.schemas import RetrievedItem, RetrievalResult


def test_evaluate_retrieval_spec_writes_metrics_without_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mirage.artifacts.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "mirage.evaluate.load_eval_examples_for_spec",
        lambda spec: [
            type("Example", (), {"qid": "q1", "question": "alpha", "gold_doc_ids": ["doc-a"]})(),
            type("Example", (), {"qid": "q2", "question": "beta", "gold_doc_ids": ["doc-b"]})(),
        ],
    )

    def fake_policy(spec, question: str, qid: str | None):
        doc_id = "doc-a" if qid == "q1" else "doc-x"
        retrieval = RetrievalResult(
            qid=qid,
            question=question,
            retrieved_doc_ids=[doc_id],
            query_embedding_tokens=3,
            query_embedding_cost_usd=0.000001,
            retrieval_latency_ms=10.0,
        )
        item = RetrievedItem(
            chunk_id=f"{doc_id}::chunk-0000",
            doc_id=doc_id,
            title="title",
            source="source",
            text="text",
            score=1.0,
            order=0,
        )
        return retrieval, [item]

    monkeypatch.setattr("mirage.evaluate._apply_tool_policy", fake_policy)
    spec = load_experiment_specs("studies/rag-foundation")[0]

    result = evaluate_retrieval_spec(spec)

    assert result["examples"] == 2
    assert result["hit_at_k"] == 0.5
    metrics_path = Path(result["retrieval_dir"]) / "retrieval_metrics.json"
    rows_path = Path(result["retrieval_dir"]) / "retrieval.jsonl"
    assert metrics_path.exists()
    assert rows_path.exists()
    assert not (Path(result["retrieval_dir"]) / "answers.jsonl").exists()


def test_run_retrieval_eval_deduplicates_generation_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_evaluate(spec, reset: bool = False):
        calls.append(spec.generation_model_id)
        return {"generation_model_id": spec.generation_model_id}

    monkeypatch.setattr("mirage.runner.evaluate_retrieval_spec", fake_evaluate)

    result = run_retrieval_eval(Path("studies/rag-foundation"))

    assert result["resolved_specs"] == 3
    assert result["evaluated_retrieval_specs"] == 1
    assert len(calls) == 1
