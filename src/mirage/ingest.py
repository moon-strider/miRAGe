from __future__ import annotations

import uuid
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient, models

from mirage.adapters import load_documents_for_spec
from mirage.artifacts import ArtifactLayout
from mirage.chunking import SemanticChunker, SentenceChunker, TokenChunker
from mirage.config import ChunkingConfig, ResolvedSpec
from mirage.embeddings import embed_texts
from mirage.faiss_store import build_faiss_index, load_faiss_index, save_faiss_index
from mirage.io_utils import read_json, read_jsonl, write_json, write_jsonl
from mirage.pipeline import _semantic_sentence_embedder
from mirage.preprocessing import apply_preprocessing
from mirage.schemas import Chunk, Document

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


def _make_client(spec: ResolvedSpec) -> QdrantClient:
    if spec.store_backend_kind != "qdrant":
        raise NotImplementedError(
            f"Store backend '{spec.store_backend_id}' is scaffolded but not implemented in runtime yet."
        )
    if spec.store_backend_runtime_status != "active":
        raise NotImplementedError(
            f"Store backend '{spec.store_backend_id}' is marked as {spec.store_backend_runtime_status}."
        )
    return QdrantClient(url=spec.env.qdrant_url)


def _faiss_index_path(spec: ResolvedSpec) -> Path:
    return ArtifactLayout(spec).store_dir() / "index.faiss"


def _build_chunker(spec: ResolvedSpec) -> TokenChunker | SentenceChunker | SemanticChunker:
    config = ChunkingConfig(
        kind=spec.chunking_kind,
        tokenizer=spec.chunk_tokenizer,
        chunk_size=spec.chunk_size,
        chunk_overlap=spec.chunk_overlap,
        chunking_model_id=spec.chunking_model_id,
        semantic_similarity_threshold=spec.semantic_similarity_threshold,
        semantic_min_sentences_per_chunk=spec.semantic_min_sentences_per_chunk,
    )
    if spec.chunking_kind == "semantic":
        return SemanticChunker(config, embedder=_semantic_sentence_embedder(spec))
    if spec.chunking_kind == "sentence":
        return SentenceChunker(config)
    return TokenChunker(config)


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


def _create_collection(client: QdrantClient, spec: ResolvedSpec, vector_size: int, reset: bool) -> None:
    if spec.store_index_runtime_status != "active":
        raise NotImplementedError(
            f"Store index variant '{spec.store_index_variant_id}' is marked as {spec.store_index_runtime_status}."
        )
    vector_params = models.VectorParams(
        size=vector_size,
        distance=_DISTANCE_MAP[spec.store_index_distance],
    )
    collection_name = ArtifactLayout(spec).collection_name()
    if reset:
        client.recreate_collection(collection_name=collection_name, vectors_config=vector_params)
        return
    if client.collection_exists(collection_name=collection_name):
        return
    client.create_collection(collection_name=collection_name, vectors_config=vector_params)


def _store_faiss_index(
    spec: ResolvedSpec,
    *,
    vectors: list[list[float]],
    reset: bool,
) -> None:
    if spec.store_index_runtime_status != "active":
        raise NotImplementedError(
            f"Store index variant '{spec.store_index_variant_id}' is marked as {spec.store_index_runtime_status}."
        )
    index_path = _faiss_index_path(spec)
    if index_path.exists() and not reset:
        return
    index = build_faiss_index(
        spec.store_index_kind,
        vectors,
        nlist=getattr(spec, "store_index_nlist", None),
        m=getattr(spec, "store_index_m", None),
        bits=getattr(spec, "store_index_bits", None),
    )
    save_faiss_index(index, index_path)


def prepare_documents(spec: ResolvedSpec, *, reset: bool = False) -> tuple[list[Document], dict[str, object]]:
    layout = ArtifactLayout(spec)
    prepared_dir = layout.prepared_dir()
    prepared_path = prepared_dir / "documents.jsonl"

    if not reset and prepared_path.exists():
        documents = read_jsonl(prepared_path, Document)
        if documents:
            return documents, {
                "prepared_dir": str(prepared_dir),
                "documents": len(documents),
                "reused_prepared": True,
            }

    documents = load_documents_for_spec(spec)
    if not documents:
        raise ValueError(f"No documents found for dataset adapter '{spec.dataset_adapter_id}'")
    documents = apply_preprocessing(spec.preprocessing_kind, documents)
    if not documents:
        raise ValueError(f"Preprocessing variant '{spec.preprocessing_variant_id}' produced zero documents")

    prepared_rows = [document.model_dump(mode="json") for document in documents]
    write_jsonl(prepared_path, prepared_rows)
    write_json(prepared_dir / "resolved_spec.json", spec.model_dump(mode="json", exclude={"env"}))
    return documents, {
        "prepared_dir": str(prepared_dir),
        "documents": len(documents),
        "reused_prepared": False,
    }


def build_chunks(
    spec: ResolvedSpec,
    documents: list[Document],
    *,
    reset: bool = False,
) -> tuple[list[Chunk], dict[str, object]]:
    layout = ArtifactLayout(spec)
    chunks_dir = layout.chunks_dir()
    chunks_path = chunks_dir / "chunks.jsonl"

    if not reset and chunks_path.exists():
        chunks = read_jsonl(chunks_path, Chunk)
        if chunks:
            return chunks, {
                "chunks_dir": str(chunks_dir),
                "chunks": len(chunks),
                "reused_chunks": True,
            }

    chunker = _build_chunker(spec)
    chunks = chunker.chunk_documents(documents)
    if not chunks:
        raise ValueError("Chunking produced zero chunks")

    write_jsonl(chunks_path, [chunk.model_dump(mode="json") for chunk in chunks])
    write_json(chunks_dir / "resolved_spec.json", spec.model_dump(mode="json", exclude={"env"}))
    return chunks, {
        "chunks_dir": str(chunks_dir),
        "chunks": len(chunks),
        "reused_chunks": False,
    }


def ingest_spec(spec: ResolvedSpec, reset: bool = False) -> dict[str, object]:
    started = perf_counter()
    documents, prepared_info = prepare_documents(spec, reset=reset)
    chunks, chunk_info = build_chunks(spec, documents, reset=reset)

    layout = ArtifactLayout(spec)
    store_dir = layout.store_dir()
    metadata_path = store_dir / "metadata.json"
    if spec.store_backend_kind == "qdrant":
        client = _make_client(spec)
        collection_name = layout.collection_name()
        if not reset and metadata_path.exists() and client.collection_exists(collection_name=collection_name):
            metadata = read_json(metadata_path)
            return {
                **metadata,
                "prepared_dir": prepared_info["prepared_dir"],
                "chunks_dir": chunk_info["chunks_dir"],
                "store_dir": str(store_dir),
                "reused_prepared": prepared_info["reused_prepared"],
                "reused_chunks": chunk_info["reused_chunks"],
                "reused_store": True,
            }
    elif spec.store_backend_kind == "faiss":
        if not reset and metadata_path.exists() and _faiss_index_path(spec).exists():
            metadata = read_json(metadata_path)
            return {
                **metadata,
                "prepared_dir": prepared_info["prepared_dir"],
                "chunks_dir": chunk_info["chunks_dir"],
                "store_dir": str(store_dir),
                "reused_prepared": prepared_info["reused_prepared"],
                "reused_chunks": chunk_info["reused_chunks"],
                "reused_store": True,
            }
    else:
        raise NotImplementedError(
            f"Store backend '{spec.store_backend_id}' is scaffolded but not implemented in runtime yet."
        )

    vectors, embedding_tokens, embedding_cost_usd = embed_texts(
        provider=spec.store_embedding_provider,
        model=spec.store_embedding_model,
        texts=[chunk.text for chunk in chunks],
        batch_size=spec.store_embedding_batch_size,
        env=spec.env,
        pricing_input_per_1m_tokens_usd=spec.store_embedding_pricing_input_per_1m_tokens_usd,
    )

    if not vectors:
        raise ValueError("Embedding produced zero vectors")

    collection_name = layout.collection_name()
    if spec.store_backend_kind == "qdrant":
        client = _make_client(spec)
        _create_collection(client, spec, len(vectors[0]), reset=reset)
        client.upsert(collection_name=collection_name, points=_build_points(chunks, vectors), wait=True)
    elif spec.store_backend_kind == "faiss":
        _store_faiss_index(spec, vectors=vectors, reset=reset)
    else:
        raise NotImplementedError(
            f"Store backend '{spec.store_backend_id}' is scaffolded but not implemented in runtime yet."
        )

    duration_ms = round((perf_counter() - started) * 1000, 2)
    metadata = {
        "group_id": spec.group_id,
        "experiment_id": spec.experiment_id,
        "dataset_id": spec.dataset_id,
        "load_variant_id": spec.load_variant_id,
        "store_variant_id": spec.store_variant_id,
        "store_backend_id": spec.store_backend_id,
        "store_index_variant_id": spec.store_index_variant_id,
        "store_embedding_model_id": spec.store_embedding_model_id,
        "collection_name": collection_name,
        "documents": prepared_info["documents"],
        "chunks": chunk_info["chunks"],
        "store_embedding_tokens": embedding_tokens,
        "build_store_embedding_cost_usd": embedding_cost_usd,
        "duration_ms": duration_ms,
    }
    write_json(store_dir / "resolved_spec.json", spec.model_dump(mode="json", exclude={"env"}))
    write_json(metadata_path, metadata)
    return {
        **metadata,
        "prepared_dir": prepared_info["prepared_dir"],
        "chunks_dir": chunk_info["chunks_dir"],
        "store_dir": str(store_dir),
        "reused_prepared": prepared_info["reused_prepared"],
        "reused_chunks": chunk_info["reused_chunks"],
        "reused_store": False,
    }
