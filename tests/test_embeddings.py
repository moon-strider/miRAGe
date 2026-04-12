from __future__ import annotations

from types import SimpleNamespace

import pytest

from mirage.embeddings import embed_texts


class _FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append((model, list(input)))
        offset = sum(len(batch) for _, batch in self.calls[:-1])
        data = [SimpleNamespace(embedding=[float(offset + index), 1.0]) for index, _ in enumerate(input)]
        usage = SimpleNamespace(prompt_tokens=len(input) * 10)
        return SimpleNamespace(data=data, usage=usage)


@pytest.mark.parametrize("batch_size,expected_batches", [(2, [2, 2, 1]), (3, [3, 2])])
def test_openrouter_embeddings_are_sent_in_batches(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    expected_batches: list[int],
) -> None:
    client = _FakeEmbeddingsClient()
    monkeypatch.setattr("mirage.embeddings._make_openrouter_client", lambda env: client)

    vectors, token_count, cost = embed_texts(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        texts=["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"],
        batch_size=batch_size,
        env=SimpleNamespace(openrouter_api_key="test", openrouter_base_url="https://openrouter.ai/api/v1"),
        pricing_input_per_1m_tokens_usd=0.02,
    )

    assert [len(batch) for _, batch in client.calls] == expected_batches
    assert all(model == "openai/text-embedding-3-small" for model, _ in client.calls)
    assert len(vectors) == 5
    assert token_count == 50
    assert cost == 0.000001
