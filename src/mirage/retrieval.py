from __future__ import annotations

from collections import defaultdict
import math

from rank_bm25 import BM25Okapi

from mirage.schemas import RetrievedItem


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def sparse_search(question: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]:
    if not items:
        return []
    corpus = [_tokenize(item.text) for item in items]
    model = BM25Okapi(corpus)
    query_tokens = _tokenize(question)
    scores = model.get_scores(query_tokens)
    ranked_pairs = sorted(zip(items, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    ranked: list[RetrievedItem] = []
    for order, (item, score) in enumerate(ranked_pairs[:top_k]):
        ranked.append(item.model_copy(update={"score": float(score), "order": order}))
    return ranked


def mmr_search(
    query_vector: list[float],
    items: list[RetrievedItem],
    item_vectors: list[list[float]],
    *,
    top_k: int,
    lambda_weight: float,
) -> list[RetrievedItem]:
    if not items:
        return []
    if len(items) != len(item_vectors):
        raise ValueError("MMR requires one vector per item")
    candidate_ids = list(range(len(items)))
    selected: list[int] = []
    while candidate_ids and len(selected) < top_k:
        best_idx = candidate_ids[0]
        best_score = float("-inf")
        for idx in candidate_ids:
            relevance = _cosine_similarity(query_vector, item_vectors[idx])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine_similarity(item_vectors[idx], item_vectors[selected_idx]) for selected_idx in selected
                )
            score = lambda_weight * relevance - (1.0 - lambda_weight) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        candidate_ids.remove(best_idx)
    return [
        items[idx].model_copy(update={"score": float(_cosine_similarity(query_vector, item_vectors[idx])), "order": order})
        for order, idx in enumerate(selected)
    ]


def rrf_fuse(rankings: list[list[RetrievedItem]], top_k: int, rrf_k: int = 60) -> list[RetrievedItem]:
    fused_scores: dict[str, float] = defaultdict(float)
    canonical: dict[str, RetrievedItem] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused_scores[item.chunk_id] += 1.0 / (rrf_k + rank)
            canonical[item.chunk_id] = item
    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    fused: list[RetrievedItem] = []
    for order, (chunk_id, score) in enumerate(ordered[:top_k]):
        fused.append(canonical[chunk_id].model_copy(update={"score": float(score), "order": order}))
    return fused
