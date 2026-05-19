from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from time import sleep

import httpx
import tiktoken
from fastembed import TextEmbedding
from openai import OpenAI

from mirage.config import EnvironmentSettings

_ENCODING = tiktoken.get_encoding("cl100k_base")
_RETRYABLE_ERROR_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def estimate_token_count(texts: Iterable[str]) -> int:
    total = 0
    for text in texts:
        total += len(_ENCODING.encode(text))
    return total


def _make_openrouter_client(env: EnvironmentSettings) -> OpenAI:
    if not env.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for remote embedding or generation")
    http_client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0))
    return OpenAI(api_key=env.openrouter_api_key, base_url=env.openrouter_base_url, http_client=http_client, max_retries=0)


def _extract_error_code(response: object) -> int | None:
    error = getattr(response, "error", None)
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


def _create_embeddings_with_retries(client: OpenAI, *, model: str, batch: list[str], max_attempts: int = 6) -> object:
    last_error: object = None
    for attempt in range(max_attempts):
        try:
            response = client.embeddings.create(model=model, input=batch)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Embedding request failed for {model}: {last_error}") from exc
            sleep(min(2**attempt, 30))
            continue
        data = getattr(response, "data", None)
        if data is not None:
            return response
        error_code = _extract_error_code(response)
        last_error = getattr(response, "error", None)
        if error_code not in _RETRYABLE_ERROR_CODES or attempt == max_attempts - 1:
            raise RuntimeError(f"Embedding request failed for {model}: {last_error}")
        sleep(min(2**attempt, 30))
    raise RuntimeError(f"Embedding request failed for {model}: {last_error}")


def _safe_cache_segment(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in value).strip("-") or "default"


def _embedding_cache_path(cache_dir: Path, *, cache_namespace: str, provider: str, model: str, text: str) -> Path:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    namespace = Path(*[_safe_cache_segment(part) for part in cache_namespace.split("/") if part])
    model_key = _safe_cache_segment(f"{provider}-{model}")
    return cache_dir / namespace / model_key / f"{text_hash}.json"


def _read_cached_vector(path: Path, *, provider: str, model: str, text: str) -> tuple[list[float], int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("provider") != provider or payload.get("model") != model:
        return None
    if payload.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
        return None
    vector = payload.get("vector")
    if not isinstance(vector, list):
        return None
    return [float(value) for value in vector], int(payload.get("token_count", estimate_token_count([text])))


def _write_cached_vector(path: Path, *, provider: str, model: str, text: str, vector: list[float], token_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "model": model,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": token_count,
        "dimensions": len(vector),
        "vector": vector,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _assigned_token_counts(texts: list[str], token_count: int) -> list[int]:
    estimated_tokens = [estimate_token_count([text]) for text in texts]
    estimated_total = sum(estimated_tokens) or len(texts)
    assigned: list[int] = []
    remainder = token_count
    for position, estimated in enumerate(estimated_tokens):
        if position == len(estimated_tokens) - 1:
            assigned.append(remainder)
            continue
        count = round((estimated / estimated_total) * token_count)
        assigned.append(count)
        remainder -= count
    return assigned


def _embed_uncached_texts(
    *,
    provider: str,
    model: str,
    texts: list[str],
    batch_size: int,
    env: EnvironmentSettings,
) -> tuple[list[list[float]], int]:
    if provider == "fastembed":
        embedder = TextEmbedding(model_name=model)
        return [vector.tolist() for vector in embedder.embed(texts, batch_size=batch_size)], 0

    if provider == "openrouter":
        client = _make_openrouter_client(env)
        vectors: list[list[float]] = []
        token_count = 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = _create_embeddings_with_retries(client, model=model, batch=batch)
            vectors.extend(item.embedding for item in response.data)
            usage = getattr(response, "usage", None)
            if usage and getattr(usage, "prompt_tokens", None) is not None:
                token_count += int(usage.prompt_tokens)
            else:
                token_count += estimate_token_count(batch)
        return vectors, token_count

    raise NotImplementedError(f"Unsupported embedding provider: {provider}")


def _embed_and_cache_openrouter_missing(
    *,
    model: str,
    missing_indexes: list[int],
    missing_texts: list[str],
    missing_paths: list[Path],
    batch_size: int,
    env: EnvironmentSettings,
    vectors_by_index: dict[int, list[float]],
    tokens_by_index: dict[int, int],
) -> None:
    client = _make_openrouter_client(env)
    for start in range(0, len(missing_texts), batch_size):
        batch = missing_texts[start : start + batch_size]
        batch_indexes = missing_indexes[start : start + batch_size]
        batch_paths = missing_paths[start : start + batch_size]
        response = _create_embeddings_with_retries(client, model=model, batch=batch)
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "prompt_tokens", None) is not None:
            token_count = int(usage.prompt_tokens)
        else:
            token_count = estimate_token_count(batch)
        assigned_tokens = _assigned_token_counts(batch, token_count)
        data = getattr(response, "data")
        for index, path, text, item, token_count_for_text in zip(
            batch_indexes, batch_paths, batch, data, assigned_tokens, strict=True
        ):
            vector_values = [float(value) for value in item.embedding]
            _write_cached_vector(
                path, provider="openrouter", model=model, text=text, vector=vector_values, token_count=token_count_for_text
            )
            vectors_by_index[index] = vector_values
            tokens_by_index[index] = token_count_for_text


def embed_texts(
    *,
    provider: str,
    model: str,
    texts: list[str],
    batch_size: int,
    env: EnvironmentSettings,
    pricing_input_per_1m_tokens_usd: float,
    cache_dir: Path | None = None,
    cache_namespace: str = "default",
) -> tuple[list[list[float]], int, float]:
    if not texts:
        return [], 0, 0.0

    if cache_dir is None:
        vectors, token_count = _embed_uncached_texts(provider=provider, model=model, texts=texts, batch_size=batch_size, env=env)
        if provider == "openrouter":
            cost = round((token_count / 1_000_000) * pricing_input_per_1m_tokens_usd, 6)
            return vectors, token_count, cost
        return vectors, 0, 0.0

    vectors_by_index: dict[int, list[float]] = {}
    tokens_by_index: dict[int, int] = {}
    missing_indexes: list[int] = []
    missing_texts: list[str] = []
    missing_paths: list[Path] = []

    for index, text in enumerate(texts):
        path = _embedding_cache_path(cache_dir, cache_namespace=cache_namespace, provider=provider, model=model, text=text)
        cached = _read_cached_vector(path, provider=provider, model=model, text=text)
        if cached is None:
            missing_indexes.append(index)
            missing_texts.append(text)
            missing_paths.append(path)
        else:
            vector, cached_token_count = cached
            vectors_by_index[index] = vector
            tokens_by_index[index] = cached_token_count

    if missing_texts:
        if provider == "openrouter":
            _embed_and_cache_openrouter_missing(
                model=model,
                missing_indexes=missing_indexes,
                missing_texts=missing_texts,
                missing_paths=missing_paths,
                batch_size=batch_size,
                env=env,
                vectors_by_index=vectors_by_index,
                tokens_by_index=tokens_by_index,
            )
        else:
            new_vectors, new_token_count = _embed_uncached_texts(
                provider=provider, model=model, texts=missing_texts, batch_size=batch_size, env=env
            )
            assigned_tokens = _assigned_token_counts(missing_texts, new_token_count)
            for index, path, text, vector, token_count_for_text in zip(
                missing_indexes, missing_paths, missing_texts, new_vectors, assigned_tokens, strict=True
            ):
                vector_values = [float(value) for value in vector]
                _write_cached_vector(
                    path, provider=provider, model=model, text=text, vector=vector_values, token_count=token_count_for_text
                )
                vectors_by_index[index] = vector_values
                tokens_by_index[index] = token_count_for_text

    vectors = [vectors_by_index[index] for index in range(len(texts))]
    token_count = sum(tokens_by_index.values())
    if provider == "openrouter":
        cost = round((token_count / 1_000_000) * pricing_input_per_1m_tokens_usd, 6)
        return vectors, token_count, cost
    return vectors, 0, 0.0
