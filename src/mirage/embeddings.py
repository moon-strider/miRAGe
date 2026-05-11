from __future__ import annotations

from collections.abc import Iterable

import httpx
import tiktoken
from fastembed import TextEmbedding
from openai import OpenAI

from mirage.config import EnvironmentSettings

_ENCODING = tiktoken.get_encoding("cl100k_base")


def estimate_token_count(texts: Iterable[str]) -> int:
    total = 0
    for text in texts:
        total += len(_ENCODING.encode(text))
    return total


def _make_openrouter_client(env: EnvironmentSettings) -> OpenAI:
    if not env.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for remote embedding or generation")
    http_client = httpx.Client(follow_redirects=True)
    return OpenAI(api_key=env.openrouter_api_key, base_url=env.openrouter_base_url, http_client=http_client)


def embed_texts(
    *,
    provider: str,
    model: str,
    texts: list[str],
    batch_size: int,
    env: EnvironmentSettings,
    pricing_input_per_1m_tokens_usd: float,
) -> tuple[list[list[float]], int, float]:
    if not texts:
        return [], 0, 0.0

    if provider == "fastembed":
        embedder = TextEmbedding(model_name=model)
        vectors = [vector.tolist() for vector in embedder.embed(texts, batch_size=batch_size)]
        return vectors, 0, 0.0

    if provider == "openrouter":
        client = _make_openrouter_client(env)
        vectors: list[list[float]] = []
        token_count = 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            usage = getattr(response, "usage", None)
            if usage and getattr(usage, "prompt_tokens", None) is not None:
                token_count += int(usage.prompt_tokens)
            else:
                token_count += estimate_token_count(batch)
        cost = round((token_count / 1_000_000) * pricing_input_per_1m_tokens_usd, 6)
        return vectors, token_count, cost

    raise NotImplementedError(f"Unsupported embedding provider: {provider}")
