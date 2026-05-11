from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from mirage.datasets import fetch_dataset, list_dataset_ids


@pytest.fixture()
def dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr("mirage.datasets.DATASETS_ROOT", root)
    return root


def test_list_dataset_ids_ignores_directories_without_source(dataset_root: Path) -> None:
    (dataset_root / "scifact").mkdir()
    (dataset_root / "scifact" / "source.json").write_text("{}", encoding="utf-8")
    (dataset_root / "tmp").mkdir()

    assert list_dataset_ids() == ["scifact"]


def test_fetch_dataset_downloads_and_extracts_zip(dataset_root: Path, tmp_path: Path) -> None:
    archive_path = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive_path, "w") as handle:
        handle.writestr("scifact/corpus.jsonl", "{}\n")
        handle.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\n")

    dataset_dir = dataset_root / "scifact"
    dataset_dir.mkdir()
    (dataset_dir / "source.json").write_text(
        json.dumps(
            {
                "source_urls": {"archive": archive_path.as_uri()},
                "pinned": {"archive_md5": hashlib.md5(archive_path.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )

    result = fetch_dataset("scifact")

    assert result["dataset_id"] == "scifact"
    assert result["archives"] == [str(dataset_dir / "downloads" / "scifact.zip")]
    assert (dataset_dir / "raw" / "scifact" / "corpus.jsonl").exists()
    assert (dataset_dir / "raw" / "scifact" / "qrels" / "test.tsv").exists()


def test_fetch_dataset_downloads_and_extracts_multiple_archives(dataset_root: Path, tmp_path: Path) -> None:
    train_dev_path = tmp_path / "qasper-train-dev-v0.3.tgz"
    with tarfile.open(train_dev_path, "w:gz") as handle:
        train_path = tmp_path / "qasper-train-v0.3.json"
        train_path.write_text("{}", encoding="utf-8")
        dev_path = tmp_path / "qasper-dev-v0.3.json"
        dev_path.write_text("{}", encoding="utf-8")
        handle.add(train_path, arcname="qasper-train-v0.3.json")
        handle.add(dev_path, arcname="qasper-dev-v0.3.json")

    test_path = tmp_path / "qasper-test-and-evaluator-v0.3.tgz"
    with tarfile.open(test_path, "w:gz") as handle:
        raw_test_path = tmp_path / "qasper-test-v0.3.json"
        raw_test_path.write_text("{}", encoding="utf-8")
        handle.add(raw_test_path, arcname="qasper-test-v0.3.json")

    dataset_dir = dataset_root / "qasper"
    dataset_dir.mkdir()
    (dataset_dir / "source.json").write_text(
        json.dumps(
            {
                "source_urls": {
                    "train_dev_archive": train_dev_path.as_uri(),
                    "test_archive": test_path.as_uri(),
                }
            }
        ),
        encoding="utf-8",
    )

    result = fetch_dataset("qasper")

    assert result["dataset_id"] == "qasper"
    assert len(result["archives"]) == 2
    assert (dataset_dir / "raw" / "qasper-train-v0.3.json").exists()
    assert (dataset_dir / "raw" / "qasper-dev-v0.3.json").exists()
    assert (dataset_dir / "raw" / "qasper-test-v0.3.json").exists()
