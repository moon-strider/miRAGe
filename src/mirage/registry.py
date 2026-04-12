from __future__ import annotations

import ast
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_toml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def load_toml_dir(path: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for file_path in sorted(path.glob("*.toml")):
        merged = deep_merge(merged, load_toml_file(file_path))
    return merged


def parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return "none"
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def parse_cli_overrides(overrides: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        parsed[key.strip()] = parse_override_value(raw_value.strip())
    return parsed
