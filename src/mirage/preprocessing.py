from __future__ import annotations

import json
import re

from mirage.schemas import Document

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _clean_document(document: Document) -> Document:
    metadata = {
        key: value
        for key, value in document.metadata.items()
        if value not in (None, "", [], {}, ())
    }
    return document.model_copy(
        update={
            "title": _clean_text(document.title),
            "source": _clean_text(document.source),
            "text": _clean_text(document.text),
            "metadata": metadata,
        }
    )


def _metadata_header(document: Document) -> str:
    metadata_json = json.dumps(document.metadata, ensure_ascii=False, sort_keys=True) if document.metadata else "{}"
    return "\n".join(
        [
            f"Title: {document.title}",
            f"Source: {document.source}",
            f"Metadata: {metadata_json}",
            "Content:",
        ]
    )


def apply_preprocessing(kind: str, documents: list[Document]) -> list[Document]:
    cleaned = [_clean_document(document) for document in documents]
    if kind == "basic-clean":
        return cleaned
    if kind == "basic-clean-dedupe":
        deduped: list[Document] = []
        seen_keys: set[tuple[str, str]] = set()
        for document in cleaned:
            dedupe_key = (document.title.casefold(), document.text.casefold())
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped.append(document)
        return deduped
    if kind == "basic-clean-metadata":
        return [
            document.model_copy(update={"text": _metadata_header(document) + "\n" + document.text})
            for document in cleaned
        ]
    raise ValueError(f"Unsupported preprocessing kind: {kind}")
