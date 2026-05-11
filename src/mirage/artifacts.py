from __future__ import annotations

from pathlib import Path
import re

from mirage.config import ResolvedSpec
from mirage.registry import PROJECT_ROOT

_SAFE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_segment(value: str) -> str:
    sanitized = _SAFE_SEGMENT_RE.sub("-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "default"


class ArtifactLayout:
    def __init__(self, spec: ResolvedSpec) -> None:
        self.spec = spec
        self.root = PROJECT_ROOT / spec.artifacts_dir

    def prepared_dir(self) -> Path:
        return self.root / "prepared" / self.spec.dataset_id / self.spec.preprocessing_variant_id

    def chunks_dir(self) -> Path:
        return (
            self.root
            / "chunks"
            / self.spec.dataset_id
            / self.spec.preprocessing_variant_id
            / self.spec.chunking_variant_id
            / self.spec.chunking_model_id
        )

    def store_dir(self) -> Path:
        return self.root / "store" / self.spec.dataset_id / self.spec.load_variant_id / self.spec.store_variant_id

    def retrieval_dir(self) -> Path:
        return (
            self.root
            / "retrieval"
            / self.spec.dataset_id
            / self.spec.evalset_id
            / self.spec.load_variant_id
            / self.spec.store_variant_id
            / self.spec.query_embedding_model_id
            / self.spec.search_algorithm_id
            / self.spec.reranker_id
            / self.spec.tool_policy_id
        )

    def answers_dir(self) -> Path:
        return (
            self.root
            / "answers"
            / self.spec.dataset_id
            / self.spec.evalset_id
            / self.spec.load_variant_id
            / self.spec.store_variant_id
            / self.spec.inference_variant_id
            / self.spec.generation_model_id
        )

    def collection_name(self) -> str:
        prefix = _safe_segment(self.spec.store_collection_prefix)
        load_part = _safe_segment(self.spec.load_variant_id)
        store_part = _safe_segment(self.spec.store_variant_id)
        value = f"{prefix}-{load_part}-{store_part}"
        return value[:200]
