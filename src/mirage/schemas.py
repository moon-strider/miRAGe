from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    source: str
    text: str
    lang: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    order: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalExample(BaseModel):
    qid: str
    question: str
    gold_answers: list[str]
    gold_doc_ids: list[str]
    answer_type: str = "factoid"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedItem(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    score: float
    order: int


class RetrievalResult(BaseModel):
    qid: str | None = None
    question: str
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    query_embedding_tokens: int = 0
    query_embedding_cost_usd: float = 0.0
    retrieval_latency_ms: float = 0.0


class AnswerResult(BaseModel):
    qid: str | None = None
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    generation_input_cost_usd: float = 0.0
    generation_output_cost_usd: float = 0.0


class RunMetrics(BaseModel):
    retrieval: dict[str, float]
    generation: dict[str, float]
    operational: dict[str, float]
    cost: dict[str, float]
