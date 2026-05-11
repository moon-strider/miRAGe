from __future__ import annotations

from types import SimpleNamespace

from mirage.config import load_experiment_specs
from mirage.pipeline import answer_question
from mirage.schemas import RetrievedItem, RetrievalResult


def _item(chunk_id: str, text: str) -> RetrievedItem:
    return RetrievedItem(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::", 1)[0],
        title=chunk_id,
        source="seed://test",
        text=text,
        score=1.0,
        order=0,
    )


class _CapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    class chat:
        completions = None


def test_evidence_first_prompt_includes_evidence_instruction(monkeypatch) -> None:
    spec = load_experiment_specs(
        "experiments/01-rag-foundation/inference-prompting",
        overrides={
            "generation_model_id": "gen-llama-3.1-8b",
            "prompt_variant_id": "prompt-evidence-first-v1",
        },
    )[0]
    client = _CapturingClient()

    def create(**kwargs):
        client.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Evidence:\n[doc-001]\nAnswer:\nalpha [doc-001]"))],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=7),
        )

    client.chat.completions = SimpleNamespace(create=create)

    monkeypatch.setattr(
        "mirage.pipeline._apply_tool_policy",
        lambda runtime_spec, question, qid: (
            RetrievalResult(
                qid=qid,
                question=question,
                retrieved_doc_ids=["doc-001"],
                query_embedding_tokens=2,
                query_embedding_cost_usd=0.0,
                retrieval_latency_ms=1.0,
            ),
            [_item("doc-001::chunk-0000", "alpha evidence")],
        ),
    )
    monkeypatch.setattr("mirage.pipeline._make_openrouter_client", lambda runtime_spec: client)

    _, answer, _ = answer_question(spec, "What is alpha?", qid="q-1")

    assert answer.citations == ["doc-001"]
    assert client.calls
    user_message = client.calls[0]["messages"][1]["content"]
    assert "Evidence:" in user_message
    assert "Answer:" in user_message


def test_strict_abstain_prompt_keeps_exact_abstention_rule(monkeypatch) -> None:
    spec = load_experiment_specs(
        "experiments/01-rag-foundation/inference-prompting",
        overrides={
            "generation_model_id": "gen-llama-3.1-8b",
            "prompt_variant_id": "prompt-strict-abstain-v1",
        },
    )[0]
    client = _CapturingClient()

    def create(**kwargs):
        client.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=spec.insufficient_context_response))],
            usage=SimpleNamespace(prompt_tokens=6, completion_tokens=3),
        )

    client.chat.completions = SimpleNamespace(create=create)

    monkeypatch.setattr(
        "mirage.pipeline._apply_tool_policy",
        lambda runtime_spec, question, qid: (
            RetrievalResult(
                qid=qid,
                question=question,
                retrieved_doc_ids=["doc-001"],
                query_embedding_tokens=2,
                query_embedding_cost_usd=0.0,
                retrieval_latency_ms=1.0,
            ),
            [_item("doc-001::chunk-0000", "weak evidence")],
        ),
    )
    monkeypatch.setattr("mirage.pipeline._make_openrouter_client", lambda runtime_spec: client)

    _, answer, _ = answer_question(spec, "What is missing?", qid="q-2")

    assert answer.answer == spec.insufficient_context_response
    system_message = client.calls[0]["messages"][0]["content"]
    assert "reply exactly with" in system_message
    assert spec.insufficient_context_response in system_message
