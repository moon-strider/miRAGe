from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from mirage.config import ResolvedSpec
from mirage.io_utils import read_jsonl
from mirage.schemas import Document, EvalExample


def load_documents_for_spec(spec: ResolvedSpec) -> list[Document]:
    if spec.dataset_adapter_id == "jsonl":
        if not spec.corpus_path:
            raise ValueError(f"Dataset '{spec.dataset_id}' is missing corpus_path")
        return read_jsonl(spec.corpus_path, Document)
    if spec.dataset_adapter_id == "scifact":
        return _load_scifact_documents(Path(spec.dataset_root))
    if spec.dataset_adapter_id == "fiqa":
        return _load_fiqa_documents(Path(spec.dataset_root))
    if spec.dataset_adapter_id == "qasper":
        return _load_qasper_documents(Path(spec.dataset_root))
    raise ValueError(f"Unsupported dataset adapter: {spec.dataset_adapter_id}")


def load_eval_examples_for_spec(spec: ResolvedSpec) -> list[EvalExample]:
    if spec.eval_adapter_id == "jsonl":
        if not spec.eval_path:
            raise ValueError(f"Eval set '{spec.evalset_id}' is missing eval_path")
        return read_jsonl(spec.eval_path, EvalExample)
    if spec.eval_adapter_id == "scifact":
        return _load_scifact_eval_examples(Path(spec.eval_root), split=spec.eval_split)
    if spec.eval_adapter_id == "fiqa":
        return _load_fiqa_eval_examples(Path(spec.eval_root), split=spec.eval_split)
    if spec.eval_adapter_id == "qasper":
        return _load_qasper_eval_examples(Path(spec.eval_root), split=spec.eval_split)
    raise ValueError(f"Unsupported eval adapter: {spec.eval_adapter_id}")


def _read_raw_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_qrels(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping.setdefault(row["query-id"], []).append(row["corpus-id"])
    return mapping


def _collect_scifact_gold_answers(query_row: dict, gold_doc_ids: list[str]) -> list[str]:
    labels: list[str] = []
    metadata = query_row.get("metadata", {})
    for doc_id in gold_doc_ids:
        for evidence in metadata.get(str(doc_id), []):
            label = str(evidence.get("label", "")).strip().lower()
            if label and label not in labels:
                labels.append(label)
    return labels


def _load_scifact_documents(dataset_root: Path) -> list[Document]:
    rows = _read_raw_jsonl(dataset_root / "raw" / "scifact" / "corpus.jsonl")
    return [
        Document(
            doc_id=str(row["_id"]),
            title=str(row.get("title") or f"SciFact {row['_id']}"),
            source="scifact",
            text=str(row.get("text", "")),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def _load_scifact_eval_examples(dataset_root: Path, *, split: str) -> list[EvalExample]:
    queries = {
        str(row["_id"]): row
        for row in _read_raw_jsonl(dataset_root / "raw" / "scifact" / "queries.jsonl")
    }
    qrels = _read_qrels(dataset_root / "raw" / "scifact" / "qrels" / f"{split}.tsv")

    examples: list[EvalExample] = []
    for query_id, gold_doc_ids in qrels.items():
        query_row = queries[query_id]
        examples.append(
            EvalExample(
                qid=query_id,
                question=str(query_row["text"]),
                gold_answers=_collect_scifact_gold_answers(query_row, gold_doc_ids),
                gold_doc_ids=gold_doc_ids,
                answer_type="verification",
                metadata={
                    "dataset": "scifact",
                    "split": split,
                    "evidence": query_row.get("metadata", {}),
                },
            )
        )
    return examples


def _load_fiqa_documents(dataset_root: Path) -> list[Document]:
    rows = _read_raw_jsonl(dataset_root / "raw" / "fiqa" / "corpus.jsonl")
    documents: list[Document] = []
    for row in rows:
        doc_id = str(row["_id"])
        title = str(row.get("title") or "").strip() or f"FiQA {doc_id}"
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                source="fiqa",
                text=str(row.get("text", "")),
                metadata=row.get("metadata", {}),
            )
        )
    return documents


def _load_fiqa_eval_examples(dataset_root: Path, *, split: str) -> list[EvalExample]:
    queries = {
        str(row["_id"]): row
        for row in _read_raw_jsonl(dataset_root / "raw" / "fiqa" / "queries.jsonl")
    }
    qrels = _read_qrels(dataset_root / "raw" / "fiqa" / "qrels" / f"{split}.tsv")

    return [
        EvalExample(
            qid=query_id,
            question=str(queries[query_id]["text"]),
            gold_answers=[],
            gold_doc_ids=gold_doc_ids,
            answer_type="retrieval_only",
            metadata={"dataset": "fiqa", "split": split},
        )
        for query_id, gold_doc_ids in qrels.items()
    ]


def _qasper_full_text_blocks(paper: dict) -> Iterable[str]:
    abstract = str(paper.get("abstract", "")).strip()
    if abstract:
        yield "Abstract\n" + abstract

    for section in paper.get("full_text", []):
        section_name = str(section.get("section_name") or "Untitled section").strip()
        paragraphs = [str(paragraph).strip() for paragraph in section.get("paragraphs", []) if str(paragraph).strip()]
        if not paragraphs:
            continue
        yield section_name + "\n" + "\n\n".join(paragraphs)


def _load_qasper_documents(dataset_root: Path) -> list[Document]:
    documents: list[Document] = []
    raw_dir = dataset_root / "raw"
    for filename in ("qasper-train-v0.3.json", "qasper-dev-v0.3.json"):
        path = raw_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        split = "train" if "train" in filename else "dev"
        for paper_id, paper in data.items():
            text = "\n\n".join(_qasper_full_text_blocks(paper))
            documents.append(
                Document(
                    doc_id=str(paper_id),
                    title=str(paper.get("title") or paper_id),
                    source="qasper",
                    text=text,
                    metadata={
                        "dataset": "qasper",
                        "split": split,
                        "sections": [
                            str(section.get("section_name") or "Untitled section")
                            for section in paper.get("full_text", [])
                        ],
                        "figures_and_tables": len(paper.get("figures_and_tables", [])),
                    },
                )
            )
    return documents


def _qasper_answer_text(answer_payload: dict) -> str:
    if answer_payload.get("unanswerable"):
        return ""
    free_form = str(answer_payload.get("free_form_answer") or "").strip()
    if free_form:
        return free_form
    extractive_spans = [str(span).strip() for span in answer_payload.get("extractive_spans", []) if str(span).strip()]
    if extractive_spans:
        return " | ".join(extractive_spans)
    yes_no = answer_payload.get("yes_no")
    if yes_no is True:
        return "yes"
    if yes_no is False:
        return "no"
    return ""


def _load_qasper_eval_examples(dataset_root: Path, *, split: str) -> list[EvalExample]:
    path = dataset_root / "raw" / f"qasper-{split}-v0.3.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    examples: list[EvalExample] = []
    for paper_id, paper in data.items():
        for qa in paper.get("qas", []):
            answers: list[str] = []
            evidence_sets: list[list[str]] = []
            for answer_row in qa.get("answers", []):
                answer_payload = answer_row.get("answer", {})
                answer_text = _qasper_answer_text(answer_payload)
                if answer_text and answer_text not in answers:
                    answers.append(answer_text)
                evidence = [str(item).strip() for item in answer_payload.get("evidence", []) if str(item).strip()]
                if evidence:
                    evidence_sets.append(evidence)
            if not answers:
                continue
            examples.append(
                EvalExample(
                    qid=str(qa["question_id"]),
                    question=str(qa["question"]),
                    gold_answers=answers,
                    gold_doc_ids=[str(paper_id)],
                    answer_type="longform",
                    metadata={
                        "dataset": "qasper",
                        "split": split,
                        "paper_id": str(paper_id),
                        "evidence_sets": evidence_sets,
                        "question_writer": qa.get("question_writer"),
                        "nlp_background": qa.get("nlp_background"),
                        "topic_background": qa.get("topic_background"),
                        "paper_read": qa.get("paper_read"),
                    },
                )
            )
    return examples
