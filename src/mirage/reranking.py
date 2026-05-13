from __future__ import annotations

import httpx
from fastembed.rerank.cross_encoder import TextCrossEncoder

from mirage.config import EnvironmentSettings
from mirage.schemas import RetrievedItem


def _ranked_items(items: list[RetrievedItem], scores: list[float]) -> list[RetrievedItem]:
    ranked_pairs = sorted(zip(items, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [item.model_copy(update={"score": float(score), "order": order}) for order, (item, score) in enumerate(ranked_pairs)]


def _parse_rerank_scores(payload: dict, item_count: int) -> list[float]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Reranker response does not contain results")
    scores = [0.0] * item_count
    for row in rows:
        index = int(row.get("index"))
        score = row.get("relevance_score", row.get("score"))
        if score is None:
            raise ValueError("Reranker result does not contain a score")
        scores[index] = float(score)
    return scores


def _openrouter_rerank(
    *,
    model: str,
    question: str,
    items: list[RetrievedItem],
    env: EnvironmentSettings,
) -> list[RetrievedItem]:
    if not env.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter reranking")
    payload = {
        "model": model,
        "query": question,
        "documents": [item.text for item in items],
        "top_n": len(items),
    }
    headers = {"Authorization": f"Bearer {env.openrouter_api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{env.openrouter_base_url.rstrip('/')}/rerank", headers=headers, json=payload)
        response.raise_for_status()
        scores = _parse_rerank_scores(response.json(), len(items))
    return _ranked_items(items, scores)


def rerank_items(
    *,
    reranker_id: str,
    reranker_kind: str,
    reranker_model: str | None,
    reranker_batch_size: int,
    question: str,
    items: list[RetrievedItem],
    env: EnvironmentSettings,
) -> list[RetrievedItem]:
    if reranker_id == "none" or reranker_kind == "none" or not items:
        return items
    if not reranker_model:
        raise ValueError(f"Reranker '{reranker_id}' does not declare a model")
    if reranker_kind == "cross-encoder":
        encoder = TextCrossEncoder(model_name=reranker_model)
        scores = list(encoder.rerank(question, [item.text for item in items], batch_size=reranker_batch_size))
        return _ranked_items(items, scores)
    if reranker_kind == "openrouter-rerank":
        return _openrouter_rerank(model=reranker_model, question=question, items=items, env=env)
    raise NotImplementedError(f"Unsupported reranker kind: {reranker_kind}")
