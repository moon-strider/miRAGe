from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from mirage.registry import PROJECT_ROOT

DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
_ARCHIVE_URL_KEYS = ("archive", "train_dev_archive", "test_archive")


def list_dataset_ids() -> list[str]:
    dataset_ids: list[str] = []
    if not DATASETS_ROOT.exists():
        return dataset_ids
    for path in sorted(DATASETS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "source.json").exists():
            continue
        dataset_ids.append(path.name)
    return dataset_ids


def load_dataset_source(dataset_id: str) -> dict[str, Any]:
    source_path = DATASETS_ROOT / dataset_id / "source.json"
    if not source_path.exists():
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    return json.loads(source_path.read_text(encoding="utf-8"))


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        shutil.copy2(Path(urllib.request.url2pathname(parsed.path)), destination)
        return
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffixes = archive_path.suffixes
    if suffixes and suffixes[-1] == ".zip":
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(raw_dir)
        return
    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-1] in {".tgz", ".gz"}:
        with tarfile.open(archive_path, "r:*") as handle:
            handle.extractall(raw_dir, filter="data")
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def fetch_dataset(dataset_id: str, *, force: bool = False) -> dict[str, Any]:
    dataset_dir = DATASETS_ROOT / dataset_id
    source = load_dataset_source(dataset_id)
    source_urls = source.get("source_urls", {})
    pinned = source.get("pinned", {})
    downloads_dir = dataset_dir / "downloads"
    raw_dir = dataset_dir / "raw"
    archive_paths: list[str] = []
    extracted_paths: list[str] = []

    for key in _ARCHIVE_URL_KEYS:
        url = source_urls.get(key)
        if not url:
            continue
        filename = Path(urllib.parse.urlparse(url).path).name
        archive_path = downloads_dir / filename
        archive_paths.append(str(archive_path))
        if force or not archive_path.exists():
            _download_file(url, archive_path)
        expected_md5 = pinned.get("archive_md5") if key == "archive" else None
        if expected_md5 and _md5(archive_path) != expected_md5:
            raise ValueError(f"Checksum mismatch for {archive_path.name}")
        if force and raw_dir.exists():
            shutil.rmtree(raw_dir)
        _extract_archive(archive_path, raw_dir)
        extracted_paths.append(str(raw_dir))

    if not archive_paths:
        raise ValueError(f"No downloadable archives declared for dataset_id: {dataset_id}")

    return {
        "dataset_id": dataset_id,
        "archives": archive_paths,
        "raw_dir": str(raw_dir),
        "extracted_paths": extracted_paths,
    }


def fetch_all_datasets(*, force: bool = False) -> list[dict[str, Any]]:
    return [fetch_dataset(dataset_id, force=force) for dataset_id in list_dataset_ids()]
