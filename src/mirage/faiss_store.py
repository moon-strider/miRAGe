from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from mirage.schemas import Chunk, RetrievedItem


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_faiss_index(
    index_kind: str,
    vectors: list[list[float]],
    *,
    nlist: int | None = None,
    m: int | None = None,
    bits: int | None = None,
) -> faiss.Index:
    if not vectors:
        raise ValueError("Cannot build a FAISS index without vectors")
    matrix = np.asarray(vectors, dtype="float32")
    dimension = matrix.shape[1]
    normalized = _normalize_rows(matrix.copy())
    if index_kind == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized)
        return index
    if index_kind == "ivfflat":
        if not nlist:
            raise ValueError("FAISS ivfflat requires nlist")
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(normalized)
        index.add(normalized)
        index.nprobe = min(nlist, 8)
        return index
    if index_kind == "ivfpq":
        if not nlist or not m or not bits:
            raise ValueError("FAISS ivfpq requires nlist, m, and bits")
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits, faiss.METRIC_INNER_PRODUCT)
        index.train(normalized)
        index.add(normalized)
        index.nprobe = min(nlist, 8)
        return index
    raise NotImplementedError(f"Unsupported FAISS index kind: {index_kind}")


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(file_path))


def load_faiss_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(Path(path)))


def search_faiss_index(
    *,
    index: faiss.Index,
    query_vector: list[float],
    items: list[Chunk] | list[RetrievedItem],
    top_k: int,
) -> list[RetrievedItem]:
    if not items:
        return []
    query = np.asarray([query_vector], dtype="float32")
    query = _normalize_rows(query)
    scores, indices = index.search(query, top_k)
    results: list[RetrievedItem] = []
    for order, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True)):
        if idx < 0:
            continue
        item = items[int(idx)]
        if isinstance(item, Chunk):
            results.append(
                RetrievedItem(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    title=item.title,
                    source=item.source,
                    text=item.text,
                    score=float(score),
                    order=order,
                )
            )
        else:
            results.append(item.model_copy(update={"score": float(score), "order": order}))
    return results
