from __future__ import annotations

from time import perf_counter

from openai import OpenAI
from qdrant_client import QdrantClient

from mirage.artifacts import ArtifactLayout
from mirage.config import ResolvedSpec
from mirage.embeddings import embed_texts
from mirage.faiss_store import load_faiss_index, search_faiss_index
from mirage.io_utils import read_jsonl
from mirage.metrics import dedupe_preserve_order, extract_citations
from mirage.reranking import rerank_items
from mirage.retrieval import rrf_fuse, sparse_search
from mirage.schemas import AnswerResult, Chunk, RetrievedItem, RetrievalResult


def _semantic_sentence_embedder(spec: ResolvedSpec):
    def _embed(texts: list[str]) -> list[list[float]]:
        vectors, _, _ = embed_texts(
            provider=spec.semantic_embedding_provider,
            model=spec.semantic_embedding_model,
            texts=texts,
            batch_size=spec.semantic_embedding_batch_size,
            env=spec.env,
            pricing_input_per_1m_tokens_usd=spec.semantic_embedding_pricing_input_per_1m_tokens_usd,
        )
        return vectors

    return _embed


def _make_qdrant_client(spec: ResolvedSpec) -> QdrantClient:
    return QdrantClient(url=spec.env.qdrant_url)


def _make_openrouter_client(spec: ResolvedSpec) -> OpenAI:
    if not spec.env.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for remote embedding or generation")
    return OpenAI(api_key=spec.env.openrouter_api_key, base_url=spec.env.openrouter_base_url)


def _search_qdrant(
    spec: ResolvedSpec,
    query_vector: list[float],
    *,
    limit: int,
) -> list[RetrievedItem]:
    client = _make_qdrant_client(spec)
    collection_name = ArtifactLayout(spec).collection_name()
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
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


def _scroll_all_qdrant_items(spec: ResolvedSpec) -> list[RetrievedItem]:
    client = _make_qdrant_client(spec)
    collection_name = ArtifactLayout(spec).collection_name()
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=100_000,
        with_payload=True,
        with_vectors=False,
    )
    items: list[RetrievedItem] = []
    for point in points:
        payload = point.payload or {}
        items.append(
            RetrievedItem(
                chunk_id=str(payload["chunk_id"]),
                doc_id=str(payload["doc_id"]),
                title=str(payload.get("title", "")),
                source=str(payload.get("source", "")),
                text=str(payload.get("text", "")),
                score=0.0,
                order=int(payload.get("order", 0)),
            )
        )
    return items


def _search_store(
    spec: ResolvedSpec,
    query_vector: list[float],
    *,
    limit: int,
) -> list[RetrievedItem]:
    if spec.store_backend_kind == "qdrant":
        return _search_qdrant(spec, query_vector, limit=limit)
    if spec.store_backend_kind == "faiss":
        chunks_path = ArtifactLayout(spec).chunks_dir() / "chunks.jsonl"
        chunks = read_jsonl(chunks_path, Chunk)
        index = load_faiss_index(ArtifactLayout(spec).store_dir() / "index.faiss")
        return search_faiss_index(index=index, query_vector=query_vector, items=chunks, top_k=limit)
    raise NotImplementedError(f"Unsupported store backend: {spec.store_backend_kind}")


def _scroll_all_store_items(spec: ResolvedSpec) -> list[RetrievedItem]:
    if spec.store_backend_kind == "qdrant":
        return _scroll_all_qdrant_items(spec)
    if spec.store_backend_kind == "faiss":
        chunks_path = ArtifactLayout(spec).chunks_dir() / "chunks.jsonl"
        chunks = read_jsonl(chunks_path, Chunk)
        return [
            RetrievedItem(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                text=chunk.text,
                score=0.0,
                order=chunk.order,
            )
            for chunk in chunks
        ]
    raise NotImplementedError(f"Unsupported store backend: {spec.store_backend_kind}")


def retrieve(spec: ResolvedSpec, question: str, qid: str | None = None) -> tuple[RetrievalResult, list[RetrievedItem]]:
    if spec.search_kind not in {"dense", "sparse", "hybrid"}:
        raise NotImplementedError(
            f"Search algorithm '{spec.search_algorithm_id}' is scaffolded but not implemented in runtime yet."
        )
    started = perf_counter()
    vectors, query_embedding_tokens, query_embedding_cost_usd = embed_texts(
        provider=spec.query_embedding_provider,
        model=spec.query_embedding_model,
        texts=[question],
        batch_size=1,
        env=spec.env,
        pricing_input_per_1m_tokens_usd=spec.query_embedding_pricing_input_per_1m_tokens_usd,
    )
    query_vector = vectors[0]
    dense_limit = spec.search_dense_top_k or spec.top_k
    sparse_limit = spec.search_sparse_top_k or spec.top_k
    if spec.search_kind == "dense":
        results = _search_store(spec, query_vector, limit=spec.top_k)
    elif spec.search_kind == "sparse":
        results = sparse_search(question, _scroll_all_store_items(spec), top_k=spec.top_k)
    else:
        dense_results = _search_store(spec, query_vector, limit=dense_limit)
        sparse_results = sparse_search(question, _scroll_all_store_items(spec), top_k=sparse_limit)
        results = rrf_fuse([dense_results, sparse_results], top_k=spec.top_k, rrf_k=spec.search_rrf_k)
    results = rerank_items(
        reranker_id=spec.reranker_id,
        reranker_kind=spec.reranker_kind,
        reranker_model=spec.reranker_model,
        reranker_batch_size=spec.reranker_batch_size,
        question=question,
        items=results,
    )

    retrieval_latency_ms = round((perf_counter() - started) * 1000, 2)
    retrieval = RetrievalResult(
        qid=qid,
        question=question,
        retrieved_doc_ids=dedupe_preserve_order(item.doc_id for item in results),
        query_embedding_tokens=query_embedding_tokens,
        query_embedding_cost_usd=query_embedding_cost_usd,
        retrieval_latency_ms=retrieval_latency_ms,
    )
    return retrieval, results


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


def answer_question(
    spec: ResolvedSpec,
    question: str,
    qid: str | None = None,
) -> tuple[RetrievalResult, AnswerResult, list[RetrievedItem]]:
    if spec.tool_policy_id != "none":
        raise NotImplementedError(
            f"Tool policy '{spec.tool_policy_id}' is scaffolded but not implemented in runtime yet."
        )
    retrieval, retrieved_items = retrieve(spec, question, qid=qid)
    started = perf_counter()
    context = _build_context(retrieved_items)
    prompt = spec.prompt_user_template.format(question=question, context=context)

    openrouter_client = _make_openrouter_client(spec)
    response = openrouter_client.chat.completions.create(
        model=spec.generation_model,
        temperature=spec.generation_temperature,
        max_tokens=spec.generation_max_tokens,
        messages=[
            {"role": "system", "content": spec.prompt_system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    input_tokens = int(usage.prompt_tokens) if usage else 0
    output_tokens = int(usage.completion_tokens) if usage else 0
    generation_input_cost_usd = round(
        (input_tokens / 1_000_000) * spec.generation_pricing_input_per_1m_tokens_usd,
        6,
    )
    generation_output_cost_usd = round(
        (output_tokens / 1_000_000) * spec.generation_pricing_output_per_1m_tokens_usd,
        6,
    )
    citations = extract_citations(answer)
    generation_latency_ms = (perf_counter() - started) * 1000

    result = AnswerResult(
        qid=qid,
        question=question,
        answer=answer,
        citations=citations,
        retrieved_doc_ids=retrieval.retrieved_doc_ids,
        latency_ms=round(retrieval.retrieval_latency_ms + generation_latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(
            retrieval.query_embedding_cost_usd + generation_input_cost_usd + generation_output_cost_usd,
            6,
        ),
        generation_input_cost_usd=generation_input_cost_usd,
        generation_output_cost_usd=generation_output_cost_usd,
    )
    return retrieval, result, retrieved_items
