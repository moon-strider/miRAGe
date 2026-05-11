from __future__ import annotations

from pathlib import Path

import pytest

from mirage.artifacts import ArtifactLayout
from mirage.config import load_experiment_specs
from mirage.io_utils import write_json, write_jsonl
from mirage.reporting import synthesize_reports


@pytest.fixture()
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("mirage.artifacts.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("mirage.reporting.PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_eval_artifacts(spec, value: float) -> None:
    layout = ArtifactLayout(spec)
    spec_payload = spec.model_dump(mode="json", exclude={"env"})
    write_json(layout.retrieval_dir() / "resolved_spec.json", spec_payload)
    write_json(layout.answers_dir() / "resolved_spec.json", spec_payload)
    write_jsonl(
        layout.retrieval_dir() / "retrieval.jsonl",
        [
            {
                "qid": "q-001",
                "question": "What tool manages Python dependencies?",
                "gold_doc_ids": ["doc-002"],
                "retrieval": {
                    "qid": "q-001",
                    "question": "What tool manages Python dependencies?",
                    "retrieved_doc_ids": ["doc-002"],
                    "query_embedding_tokens": 12,
                    "query_embedding_cost_usd": 0.000001,
                    "retrieval_latency_ms": 10.0,
                },
                "retrieved": [
                    {
                        "chunk_id": "doc-002::chunk-0000",
                        "doc_id": "doc-002",
                        "title": "Tooling conventions",
                        "source": "seed://tooling-conventions",
                        "text": "Python dependencies are managed with uv.",
                        "score": 0.9,
                        "order": 0,
                    }
                ],
                "metrics": {
                    "hit_at_k": value,
                    "precision_at_k": value,
                    "recall_at_k": value,
                    "mrr_at_k": value,
                    "ndcg_at_k": value,
                },
            }
        ],
    )
    write_jsonl(
        layout.answers_dir() / "answers.jsonl",
        [
            {
                "qid": "q-001",
                "question": "What tool manages Python dependencies?",
                "gold_answers": ["uv"],
                "answer": {
                    "qid": "q-001",
                    "question": "What tool manages Python dependencies?",
                    "answer": "uv [doc-002]",
                    "citations": ["doc-002"],
                    "retrieved_doc_ids": ["doc-002"],
                    "latency_ms": 25.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "estimated_cost_usd": 0.00001,
                    "generation_input_cost_usd": 0.000002,
                    "generation_output_cost_usd": 0.000003,
                },
                "metrics": {
                    "exact_match": value,
                    "token_f1": value,
                    "citation_hit_rate": value,
                },
            }
        ],
    )
    write_json(
        layout.answers_dir() / "metrics.json",
        {
            "retrieval": {
                "hit_at_k": value,
                "precision_at_k": value,
                "recall_at_k": value,
                "mrr_at_k": value,
                "ndcg_at_k": value,
            },
            "generation": {
                "exact_match": value,
                "token_f1": value,
                "citation_hit_rate": value,
            },
            "operational": {
                "examples": 1.0,
                "latency_p50_ms": 25.0,
                "latency_p95_ms": 25.0,
                "reranker_coverage": 1.0,
            },
            "cost": {
                "build_preprocessing_cost_usd": 0.0,
                "build_chunking_cost_usd": 0.0,
                "build_store_embedding_cost_usd": 0.0,
                "build_total_external_cost_usd": 0.0,
                "mean_query_embedding_cost_usd_eval": 0.000001,
                "mean_generation_input_cost_usd_eval": 0.000002,
                "mean_generation_output_cost_usd_eval": 0.000003,
                "mean_query_cost_usd_eval": 0.000006,
                "projected_query_cost_1m_usd": 6.0,
                "one_time_build_cost_usd": 0.0,
                "amortized_total_cost_1m_usd": 6.0,
            },
        },
    )


def test_synthesize_reports_writes_baseline_freeze_report(artifact_root: Path) -> None:
    specs = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")
    for spec in specs:
        _write_eval_artifacts(spec, 0.8)

    report_paths = synthesize_reports(specs, reports_root=artifact_root / "reports")

    assert len(report_paths) == 1
    content = report_paths[0].read_text(encoding="utf-8")
    assert "## Goal" in content
    assert "## Compared variants" in content
    assert "## Artifact reuse" in content
    assert "## Metrics summary" in content
    assert "## Cost summary" in content
    assert "## Failure cases" in content
    assert "## Conclusion" in content


def test_synthesize_reports_writes_candidate_reports_for_variant_groups(artifact_root: Path) -> None:
    specs = load_experiment_specs("experiments/01-rag-foundation/store-embedding-models")
    values = {
        "emb-bge-small-en-v1.5": 0.5,
        "emb-text-embedding-3-small": 0.7,
        "emb-text-embedding-3-large": 0.9,
    }
    for spec in specs:
        _write_eval_artifacts(spec, values[spec.store_embedding_model_id])

    report_paths = synthesize_reports(
        specs,
        reports_root=artifact_root / "reports",
        baseline_id="emb-bge-small-en-v1.5",
    )

    assert len(report_paths) == 2
    contents = [path.read_text(encoding="utf-8") for path in report_paths]
    assert any("candidate_id: emb-text-embedding-3-small" in content for content in contents)
    assert any("candidate_id: emb-text-embedding-3-large" in content for content in contents)
    assert all("baseline_id: emb-bge-small-en-v1.5" in content for content in contents)


def test_synthesize_reports_writes_wave1_report_against_frozen_baseline(artifact_root: Path) -> None:
    baseline_specs = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")
    candidate_specs = load_experiment_specs("experiments/01-rag-foundation/wave1-dense-topk10")
    for spec in baseline_specs:
        _write_eval_artifacts(spec, 0.6)
    for spec in candidate_specs:
        _write_eval_artifacts(spec, 0.9)

    report_paths = synthesize_reports(
        candidate_specs,
        reports_root=artifact_root / "reports",
        baseline_id="baseline-freeze",
    )

    assert len(report_paths) == 1
    content = report_paths[0].read_text(encoding="utf-8")
    assert "baseline_id: baseline-freeze" in content
    assert "candidate_id: wave1-dense-topk10" in content
    assert "search-dense-topk5-v1" in content
    assert "search-dense-topk10-v1" in content
