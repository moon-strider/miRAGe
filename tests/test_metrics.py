from mirage.metrics import citation_hit_rate, exact_match, extract_citations, reciprocal_rank, token_f1


def test_exact_match_normalizes_case_and_punctuation() -> None:
    assert exact_match("UV!", ["uv"]) == 1.0


def test_token_f1_prefers_best_gold_answer() -> None:
    score = token_f1("Qdrant vector database", ["Qdrant", "Postgres"])
    assert 0.0 < score <= 1.0


def test_extract_citations_and_hit_rate() -> None:
    citations = extract_citations("Use uv [doc-002] and just [doc-002].")
    assert citations == ["doc-002"]
    assert citation_hit_rate(citations, ["doc-002"]) == 1.0


def test_reciprocal_rank_uses_first_relevant_hit() -> None:
    assert reciprocal_rank(["doc-010", "doc-002", "doc-003"], ["doc-002"], 10) == 0.5
