from __future__ import annotations

from mirage.preprocessing import apply_preprocessing
from mirage.schemas import Document


def _doc(doc_id: str, title: str, text: str, metadata: dict | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        source="seed://test",
        text=text,
        metadata=metadata or {},
    )


def test_basic_clean_normalizes_whitespace() -> None:
    documents = [
        _doc("doc-001", "  Alpha  ", "Line one\n\nLine two   ", {"section": " intro "}),
    ]

    processed = apply_preprocessing("basic-clean", documents)

    assert processed[0].title == "Alpha"
    assert processed[0].text == "Line one Line two"
    assert processed[0].metadata == {"section": " intro "}


def test_basic_clean_dedupe_removes_duplicate_content() -> None:
    documents = [
        _doc("doc-001", "Alpha", "Same text"),
        _doc("doc-002", "Alpha", "Same text"),
        _doc("doc-003", "Beta", "Different text"),
    ]

    processed = apply_preprocessing("basic-clean-dedupe", documents)

    assert [document.doc_id for document in processed] == ["doc-001", "doc-003"]


def test_basic_clean_metadata_prepends_metadata_header() -> None:
    documents = [
        _doc("doc-001", "Alpha", "Body text", {"section": "intro", "page": 1}),
    ]

    processed = apply_preprocessing("basic-clean-metadata", documents)

    assert processed[0].text.startswith("Title: Alpha\nSource: seed://test\nMetadata: {")
    assert "Content:\nBody text" in processed[0].text
