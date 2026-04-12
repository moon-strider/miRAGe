from __future__ import annotations

from time import perf_counter

from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from mirage.config import AppConfig
from mirage.metrics import dedupe_preserve_order, extract_citations
from mirage.schemas import AnswerResult, RetrievedItem


def _make_qdrant_client(config: AppConfig) -> QdrantClient:
    return QdrantClient(url=config.qdrant_url)


def _make_embedder(config: AppConfig) -> TextEmbedding:
    return TextEmbedding(model_name=config.embedding.model)


def _make_openrouter_client(config: AppConfig) -> OpenAI:
    if not config.env.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for ask/eval commands")
    return OpenAI(
        api_key=config.env.openrouter_api_key,
        base_url=config.env.openrouter_base_url,
    )


def retrieve(config: AppConfig, question: str) -> list[RetrievedItem]:
    embedder = _make_embedder(config)
    [query_vector] = [vector.tolist() for vector in embedder.embed([question], batch_size=1)]
    client = _make_qdrant_client(config)
    hits = client.search(
        collection_name=config.collection_name,
        query_vector=query_vector,
        limit=config.retrieval.top_k,
        with_payload=True,
    )
    results: list[RetrievedItem] = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            RetrievedItem(
                chunk_id=str(payload["chunk_id"]),
                doc_id=str(payload["doc_id"]),
                title=str(payload.get("title", "")),
                source=str(payload.get("source", "")),
                text=str(payload.get("text", "")),
                score=float(hit.score),
                order=int(payload.get("order", 0)),
            )
        )
    return results


def _build_context(items: list[RetrievedItem]) -> str:
    blocks: list[str] = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"Document ID: {item.doc_id}",
                    f"Title: {item.title}",
                    f"Source: {item.source}",
                    "Content:",
                    item.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _estimate_cost(config: AppConfig, input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * config.generation.pricing_input_per_1m_tokens_usd
    output_cost = (output_tokens / 1_000_000) * config.generation.pricing_output_per_1m_tokens_usd
    return round(input_cost + output_cost, 6)


def answer_question(config: AppConfig, question: str, qid: str | None = None) -> tuple[AnswerResult, list[RetrievedItem]]:
    started = perf_counter()
    retrieved_items = retrieve(config, question)
    context = _build_context(retrieved_items)

    openrouter_client = _make_openrouter_client(config)
    response = openrouter_client.chat.completions.create(
        model=config.generation.model,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        messages=[
            {"role": "system", "content": config.generation.system_prompt.strip()},
            {
                "role": "user",
                "content": (
                    "Answer the question using only the context below.\n\n"
                    f"Question: {question}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    input_tokens = int(usage.prompt_tokens) if usage else 0
    output_tokens = int(usage.completion_tokens) if usage else 0
    citations = extract_citations(answer)
    latency_ms = (perf_counter() - started) * 1000

    result = AnswerResult(
        qid=qid,
        question=question,
        answer=answer,
        citations=citations,
        retrieved_doc_ids=dedupe_preserve_order(item.doc_id for item in retrieved_items),
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=_estimate_cost(config, input_tokens, output_tokens),
    )
    return result, retrieved_items
