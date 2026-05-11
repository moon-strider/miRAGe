from __future__ import annotations

from collections import defaultdict

from rank_bm25 import BM25Okapi

from mirage.schemas import RetrievedItem


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


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
