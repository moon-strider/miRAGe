from __future__ import annotations

from mirage.artifacts import ArtifactLayout
from mirage.config import load_experiment_specs


def test_retrieval_artifact_path_includes_tool_policy() -> None:
    baseline = load_experiment_specs("studies/rag-foundation")[0]
    tool = load_experiment_specs(
        "studies/rag-foundation",
        overrides={"study_experiment_id": "tool-context-expansion"},
    )[0]

    assert ArtifactLayout(baseline).retrieval_dir() != ArtifactLayout(tool).retrieval_dir()
    assert "tool-context-expansion-v1" in str(ArtifactLayout(tool).retrieval_dir())
