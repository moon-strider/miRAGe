from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaselineConfig(BaseModel):
    name: str


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: int = 64


class ChunkingConfig(BaseModel):
    tokenizer: str = "cl100k_base"
    chunk_size: int = 500
    chunk_overlap: int = 50


class RetrievalConfig(BaseModel):
    top_k: int = 5
    mrr_depth: int = 10
    distance: str = "cosine"
    collection: str


class GenerationConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int = 350
    insufficient_context_response: str = "INSUFFICIENT_CONTEXT"
    system_prompt: str
    pricing_input_per_1m_tokens_usd: float = 0.0
    pricing_output_per_1m_tokens_usd: float = 0.0


class EvaluationConfig(BaseModel):
    runs_dir: str = "runs"
    metrics: list[str] = Field(default_factory=list)


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str | None = None


class AppConfig(BaseModel):
    baseline: BaselineConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    env: EnvironmentSettings = Field(default_factory=EnvironmentSettings)

    @property
    def collection_name(self) -> str:
        return self.env.qdrant_collection or self.retrieval.collection

    @property
    def qdrant_url(self) -> str:
        return self.env.qdrant_url


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = AppConfig.model_validate(raw)
    config.env = EnvironmentSettings()
    return config
