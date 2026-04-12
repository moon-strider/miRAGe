from __future__ import annotations

import uuid
from collections.abc import Iterable
from time import perf_counter

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from mirage.chunking import TokenChunker
from mirage.config import AppConfig
from mirage.io_utils import read_jsonl
from mirage.schemas import Chunk, Document

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


def _make_client(config: AppConfig) -> QdrantClient:
    return QdrantClient(url=config.qdrant_url)


def _make_embedder(config: AppConfig) -> TextEmbedding:
    return TextEmbedding(model_name=config.embedding.model)


def _embed_texts(embedder: TextEmbedding, texts: Iterable[str], batch_size: int) -> list[list[float]]:
    return [vector.tolist() for vector in embedder.embed(texts, batch_size=batch_size)]


def _create_collection(client: QdrantClient, config: AppConfig, vector_size: int, reset: bool) -> None:
    collection_name = config.collection_name
    vector_params = models.VectorParams(
        size=vector_size,
        distance=_DISTANCE_MAP[config.retrieval.distance],
    )
    if reset:
        client.recreate_collection(collection_name=collection_name, vectors_config=vector_params)
        return
    if client.collection_exists(collection_name=collection_name):
        return
    client.create_collection(collection_name=collection_name, vectors_config=vector_params)


def _build_points(chunks: list[Chunk], vectors: list[list[float]]) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                    "order": chunk.order,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                },
            )
        )
    return points


def ingest_corpus(config: AppConfig, input_path: str, reset: bool = False) -> dict[str, float | int | str]:
    started = perf_counter()
    documents = read_jsonl(input_path, Document)
    if not documents:
        raise ValueError(f"No documents found in {input_path}")

    chunker = TokenChunker(config.chunking)
    chunks = chunker.chunk_documents(documents)
    if not chunks:
        raise ValueError("Chunking produced zero chunks")

    embedder = _make_embedder(config)
    vectors = _embed_texts(embedder, [chunk.text for chunk in chunks], config.embedding.batch_size)

    client = _make_client(config)
    _create_collection(client, config, len(vectors[0]), reset=reset)
    client.upsert(collection_name=config.collection_name, points=_build_points(chunks, vectors), wait=True)

    elapsed_ms = (perf_counter() - started) * 1000
    return {
        "collection": config.collection_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "duration_ms": round(elapsed_ms, 2),
    }
