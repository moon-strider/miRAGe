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


class _FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                del kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer [doc-001] [doc-002]"))],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
                )


def test_context_expansion_policy_merges_followup_retrieval(monkeypatch) -> None:
    spec = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")[0].model_copy(
        update={"tool_policy_id": "tool-context-expansion-v1"}
    )
    calls: list[tuple[int, str]] = []

    def fake_retrieve(runtime_spec, question: str, qid: str | None = None):
        del qid
        calls.append((runtime_spec.top_k, question))
        if len(calls) == 1:
            return RetrievalResult(
                question=question,
                retrieved_doc_ids=["doc-001"],
                query_embedding_tokens=10,
                query_embedding_cost_usd=0.1,
                retrieval_latency_ms=5.0,
            ), [_item("doc-001::chunk-0000", "alpha")]
        return RetrievalResult(
            question=question,
            retrieved_doc_ids=["doc-001", "doc-002"],
            query_embedding_tokens=12,
            query_embedding_cost_usd=0.2,
            retrieval_latency_ms=7.0,
        ), [_item("doc-001::chunk-0000", "alpha"), _item("doc-002::chunk-0000", "beta")]

    monkeypatch.setattr("mirage.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("mirage.pipeline._make_openrouter_client", lambda spec: _FakeClient())

    retrieval, answer, items = answer_question(spec, "What happened?")

    assert len(calls) == 2
    assert calls[1][0] > calls[0][0]
    assert retrieval.retrieved_doc_ids == ["doc-001", "doc-002"]
    assert answer.retrieved_doc_ids == ["doc-001", "doc-002"]
    assert [item.doc_id for item in items] == ["doc-001", "doc-002"]



def test_react_policy_rewrites_followup_query(monkeypatch) -> None:
    spec = load_experiment_specs("experiments/01-rag-foundation/baseline-freeze")[0].model_copy(
        update={"tool_policy_id": "tool-react-v1"}
    )
    questions: list[str] = []

    def fake_retrieve(runtime_spec, question: str, qid: str | None = None):
        del runtime_spec, qid
        questions.append(question)
        if len(questions) == 1:
            return RetrievalResult(
                question=question,
                retrieved_doc_ids=["doc-001"],
                query_embedding_tokens=5,
                query_embedding_cost_usd=0.05,
                retrieval_latency_ms=3.0,
            ), [_item("doc-001::chunk-0000", "first pass")]
        return RetrievalResult(
            question=question,
            retrieved_doc_ids=["doc-003"],
            query_embedding_tokens=6,
            query_embedding_cost_usd=0.06,
            retrieval_latency_ms=4.0,
        ), [_item("doc-003::chunk-0000", "follow up")]

    monkeypatch.setattr("mirage.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("mirage.pipeline._make_openrouter_client", lambda spec: _FakeClient())

    retrieval, answer, items = answer_question(spec, "Explain the evidence")

    assert len(questions) == 2
    assert questions[1] != questions[0]
    assert "evidence" in questions[1].lower()
    assert retrieval.retrieved_doc_ids == ["doc-001", "doc-003"]
    assert answer.retrieved_doc_ids == ["doc-001", "doc-003"]
    assert [item.doc_id for item in items] == ["doc-001", "doc-003"]
