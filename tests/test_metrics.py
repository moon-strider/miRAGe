from __future__ import annotations

from mirage.metrics import ndcg_at_k, precision_at_k, reranker_coverage


def test_precision_at_k_counts_relevant_prefix_hits() -> None:
    score = precision_at_k(["doc-1", "doc-2", "doc-3"], ["doc-2", "doc-4"], 2)

    assert score == 0.5


def test_ndcg_at_k_rewards_earlier_relevant_hits() -> None:
    score = ndcg_at_k(["doc-2", "doc-9", "doc-4"], ["doc-2", "doc-4"], 3)

    assert round(score, 4) == 0.9197


def test_reranker_coverage_compares_reranked_pool_to_final_k() -> None:
    score = reranker_coverage(20, 10)

    assert score == 2.0
