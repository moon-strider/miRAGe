from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean
from typing import Iterable

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_WORD_RE = re.compile(r"\w+")


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def exact_match(answer: str, gold_answers: list[str]) -> float:
    normalized = normalize_text(answer)
    return float(any(normalized == normalize_text(gold) for gold in gold_answers))


def token_f1(answer: str, gold_answers: list[str]) -> float:
    prediction_tokens = normalize_text(answer).split()
    if not prediction_tokens:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            continue
        overlap = Counter(prediction_tokens) & Counter(gold_tokens)
        common = sum(overlap.values())
        if common == 0:
            continue
        precision = common / len(prediction_tokens)
        recall = common / len(gold_tokens)
        best = max(best, (2 * precision * recall) / (precision + recall))
    return best


def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    return float(any(doc_id in set(gold_doc_ids) for doc_id in retrieved_doc_ids[:k]))


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    retrieved = set(retrieved_doc_ids[:k])
    return len(retrieved & gold) / len(gold)


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    gold = set(gold_doc_ids)
    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in gold:
            return 1.0 / index
    return 0.0


def extract_citations(answer: str) -> list[str]:
    return dedupe_preserve_order(match.strip() for match in _CITATION_RE.findall(answer))


def citation_hit_rate(citations: list[str], gold_doc_ids: list[str]) -> float:
    return float(any(citation in set(gold_doc_ids) for citation in citations))


def mean_metric(values: list[float]) -> float:
    return mean(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
