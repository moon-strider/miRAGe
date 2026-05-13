from __future__ import annotations

from collections.abc import Callable, Iterable
import math
import re

import tiktoken

from mirage.config import ChunkingConfig
from mirage.schemas import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _make_chunk(document: Document, *, order: int, text: str, token_count: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}::chunk-{order:04d}",
        doc_id=document.doc_id,
        title=document.title,
        source=document.source,
        text=text,
        order=order,
        token_count=token_count,
        metadata=document.metadata,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Semantic chunking vectors must share the same dimensionality")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


class TokenChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._encoding = tiktoken.get_encoding(config.tokenizer)
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        tokens = self._encoding.encode(document.text)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        step = self._chunk_size - self._chunk_overlap
        for order, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self._chunk_size]
            if not window:
                continue
            text = self._encoding.decode(window).strip()
            if not text:
                continue
            chunks.append(
                _make_chunk(
                    document,
                    order=order,
                    text=text,
                    token_count=len(window),
                )
            )
            if start + self._chunk_size >= len(tokens):
                break
        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for document in documents:
            output.extend(self.chunk_document(document))
        return output


class SentenceChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._encoding = tiktoken.get_encoding(config.tokenizer)
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap

    def _split_units(self, text: str) -> list[tuple[str, int]]:
        stripped = text.strip()
        if not stripped:
            return []

        sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
        if not sentences:
            sentences = [stripped]

        units: list[tuple[str, int]] = []
        for sentence in sentences:
            tokens = self._encoding.encode(sentence)
            if not tokens:
                continue
            if len(tokens) <= self._chunk_size:
                units.append((sentence, len(tokens)))
                continue
            for start in range(0, len(tokens), self._chunk_size):
                window = tokens[start : start + self._chunk_size]
                if not window:
                    continue
                fragment = self._encoding.decode(window).strip()
                if fragment:
                    units.append((fragment, len(window)))
                if start + self._chunk_size >= len(tokens):
                    break
        return units

    def chunk_document(self, document: Document) -> list[Chunk]:
        units = self._split_units(document.text)
        if not units:
            return []

        chunks: list[Chunk] = []
        start = 0
        order = 0
        while start < len(units):
            end = start
            chunk_parts: list[str] = []
            token_count = 0
            while end < len(units):
                unit_text, unit_tokens = units[end]
                if token_count + unit_tokens > self._chunk_size and chunk_parts:
                    break
                chunk_parts.append(unit_text)
                token_count += unit_tokens
                end += 1
                if token_count >= self._chunk_size:
                    break

            chunk_text = " ".join(chunk_parts).strip()
            if chunk_text:
                chunks.append(
                    _make_chunk(
                        document,
                        order=order,
                        text=chunk_text,
                        token_count=token_count,
                    )
                )
                order += 1

            if end >= len(units):
                break
            if self._chunk_overlap <= 0:
                start = end
                continue

            overlap_tokens = 0
            next_start = end
            while next_start > start and overlap_tokens < self._chunk_overlap:
                next_start -= 1
                overlap_tokens += units[next_start][1]
            if next_start <= start:
                next_start = max(start + 1, end - 1)
            start = next_start

        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for document in documents:
            output.extend(self.chunk_document(document))
        return output


class EmbeddingBoundaryChunker:
    def __init__(
        self,
        config: ChunkingConfig,
        *,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self._config = config
        self._encoding = tiktoken.get_encoding(config.tokenizer)
        self._embedder = embedder
        self._threshold = config.semantic_similarity_threshold
        self._min_sentences = max(config.semantic_min_sentences_per_chunk, 1)

    def _split_sentences(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
        return sentences or [stripped]

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        if self._embedder is None:
            raise NotImplementedError(
                "Embedding-boundary chunking requires an embedding-backed sentence segmenter, but runtime sentence embedding integration is not configured yet."
            )
        return self._embedder(sentences)

    def chunk_document(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.text)
        if not sentences:
            return []
        vectors = self._embed_sentences(sentences)
        if len(vectors) != len(sentences):
            raise ValueError("Semantic chunking embedder must return one vector per sentence")
        chunks: list[Chunk] = []
        current_sentences = [sentences[0]]
        current_tokens = len(self._encoding.encode(sentences[0]))
        current_vector = vectors[0]
        order = 0
        for sentence, vector in zip(sentences[1:], vectors[1:], strict=True):
            sentence_tokens = len(self._encoding.encode(sentence))
            similarity = _cosine_similarity(current_vector, vector)
            should_split = (
                len(current_sentences) >= self._min_sentences
                and (similarity < self._threshold or current_tokens + sentence_tokens > self._config.chunk_size)
            )
            if should_split:
                chunks.append(
                    _make_chunk(
                        document,
                        order=order,
                        text=" ".join(current_sentences).strip(),
                        token_count=current_tokens,
                    )
                )
                order += 1
                current_sentences = [sentence]
                current_tokens = sentence_tokens
                current_vector = vector
                continue
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            current_vector = [lhs + rhs for lhs, rhs in zip(current_vector, vector, strict=True)]
        if current_sentences:
            chunks.append(
                _make_chunk(
                    document,
                    order=order,
                    text=" ".join(current_sentences).strip(),
                    token_count=current_tokens,
                )
            )
        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for document in documents:
            output.extend(self.chunk_document(document))
        return output
