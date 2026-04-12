from __future__ import annotations

from collections.abc import Iterable

import tiktoken

from mirage.config import ChunkingConfig
from mirage.schemas import Chunk, Document


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
                Chunk(
                    chunk_id=f"{document.doc_id}::chunk-{order:04d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    source=document.source,
                    text=text,
                    order=order,
                    token_count=len(window),
                    metadata=document.metadata,
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
