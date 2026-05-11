from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from mirage.schemas import RetrievedItem


def rerank_items(
    *,
    reranker_id: str,
    reranker_kind: str,
    reranker_model: str | None,
    reranker_batch_size: int,
    question: str,
    items: list[RetrievedItem],
) -> list[RetrievedItem]:
    if reranker_id == "none" or reranker_kind == "none" or not items:
        return items
    if reranker_kind != "cross-encoder":
        raise NotImplementedError(f"Unsupported reranker kind: {reranker_kind}")
    if not reranker_model:
        raise ValueError(f"Reranker '{reranker_id}' does not declare a model")
    encoder = TextCrossEncoder(model_name=reranker_model)
    scores = list(encoder.rerank(question, [item.text for item in items], batch_size=reranker_batch_size))
    ranked_pairs = sorted(zip(items, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    reranked: list[RetrievedItem] = []
    for order, (item, score) in enumerate(ranked_pairs):
        reranked.append(item.model_copy(update={"score": float(score), "order": order}))
    return reranked
