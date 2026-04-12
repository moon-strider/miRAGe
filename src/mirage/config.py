from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mirage.registry import PROJECT_ROOT, deep_merge, load_toml_dir, load_toml_file


class ChunkingConfig(BaseModel):
    kind: str = "token"
    tokenizer: str = "cl100k_base"
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_model_id: str = "none"


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qdrant_url: str = "http://localhost:6333"


class SystemPaths(BaseModel):
    runs_dir: str = "runs"


class PricingConfig(BaseModel):
    price_catalog_version: str


class GenerationDefaults(BaseModel):
    default_temperature: float = 0.0
    default_max_tokens: int = 350
    insufficient_context_response: str = "INSUFFICIENT_CONTEXT"


class SystemConfig(BaseModel):
    paths: SystemPaths
    pricing: PricingConfig
    generation: GenerationDefaults


class DatasetEntry(BaseModel):
    corpus_path: str


class EvalSetEntry(BaseModel):
    eval_path: str


class PreprocessingVariant(BaseModel):
    kind: str


class ChunkingVariant(BaseModel):
    kind: str
    tokenizer: str = "cl100k_base"
    chunk_size: int = 500
    chunk_overlap: int = 50


class EmbeddingModelEntry(BaseModel):
    provider: str
    model: str
    batch_size: int = 64
    pricing_input_per_1m_tokens_usd: float = 0.0


class GenerationModelEntry(BaseModel):
    provider: str
    model: str
    pricing_input_per_1m_tokens_usd: float = 0.0
    pricing_output_per_1m_tokens_usd: float = 0.0


class StoreBackendEntry(BaseModel):
    kind: str
    collection_prefix: str = "mirage"
    runtime_status: str = "active"


class StoreIndexVariantEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend_id: str
    kind: str
    distance: str = "cosine"
    runtime_status: str = "active"


class SearchAlgorithmEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    top_k: int
    mrr_depth: int
    distance: str = "cosine"


class PromptVariantEntry(BaseModel):
    system_prompt: str
    user_template: str


class RerankerEntry(BaseModel):
    kind: str


class ToolPolicyEntry(BaseModel):
    kind: str


class RegistryBundle(BaseModel):
    datasets: dict[str, DatasetEntry] = Field(default_factory=dict)
    evalsets: dict[str, EvalSetEntry] = Field(default_factory=dict)
    preprocessing_variants: dict[str, PreprocessingVariant] = Field(default_factory=dict)
    chunking_variants: dict[str, ChunkingVariant] = Field(default_factory=dict)
    embedding_models: dict[str, EmbeddingModelEntry] = Field(default_factory=dict)
    generation_models: dict[str, GenerationModelEntry] = Field(default_factory=dict)
    store_backends: dict[str, StoreBackendEntry] = Field(default_factory=dict)
    store_index_variants: dict[str, StoreIndexVariantEntry] = Field(default_factory=dict)
    search_algorithms: dict[str, SearchAlgorithmEntry] = Field(default_factory=dict)
    prompt_variants: dict[str, PromptVariantEntry] = Field(default_factory=dict)
    rerankers: dict[str, RerankerEntry] = Field(default_factory=dict)
    tool_policies: dict[str, ToolPolicyEntry] = Field(default_factory=dict)


class ResolvedSpec(BaseModel):
    group_id: str
    experiment_id: str
    dataset_id: str
    evalset_id: str
    preprocessing_variant_id: str
    preprocessing_kind: str
    chunking_variant_id: str
    chunking_kind: str
    chunking_model_id: str
    chunk_tokenizer: str
    chunk_size: int
    chunk_overlap: int
    store_backend_id: str
    store_backend_kind: str
    store_backend_runtime_status: str
    store_collection_prefix: str
    store_index_variant_id: str
    store_index_kind: str
    store_index_distance: str
    store_index_runtime_status: str
    store_embedding_model_id: str
    store_embedding_provider: str
    store_embedding_model: str
    store_embedding_batch_size: int
    store_embedding_pricing_input_per_1m_tokens_usd: float
    query_embedding_model_id: str
    query_embedding_provider: str
    query_embedding_model: str
    query_embedding_batch_size: int
    query_embedding_pricing_input_per_1m_tokens_usd: float
    search_algorithm_id: str
    search_kind: str
    top_k: int
    mrr_depth: int
    reranker_id: str
    prompt_variant_id: str
    prompt_system_prompt: str
    prompt_user_template: str
    tool_policy_id: str
    generation_model_id: str
    generation_provider: str
    generation_model: str
    generation_temperature: float
    generation_max_tokens: int
    generation_pricing_input_per_1m_tokens_usd: float
    generation_pricing_output_per_1m_tokens_usd: float
    insufficient_context_response: str
    price_catalog_version: str
    runs_dir: str
    corpus_path: str
    eval_path: str
    load_variant_id: str
    store_variant_id: str
    inference_variant_id: str
    env: EnvironmentSettings = Field(default_factory=EnvironmentSettings)


def _load_system_config() -> SystemConfig:
    raw = load_toml_file(PROJECT_ROOT / "configs" / "system.toml")
    return SystemConfig.model_validate(raw)


def _load_registry_bundle() -> RegistryBundle:
    raw = load_toml_dir(PROJECT_ROOT / "configs" / "registries")
    return RegistryBundle.model_validate(raw)


def _load_prototype(family: str, prototype_id: str) -> dict[str, Any]:
    prototype_path = PROJECT_ROOT / "configs" / "prototypes" / family / f"{prototype_id}.toml"
    return load_toml_file(prototype_path)


def _load_experiment_files(experiment_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = Path(experiment_path)
    experiment_dir = path if path.is_dir() else path.parent
    group_dir = experiment_dir.parent
    group_data = load_toml_file(group_dir / "group.toml")
    overlay_data = load_toml_file(experiment_dir / "overlay.toml")
    return group_data, overlay_data, experiment_dir


def _materialize_matrix(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [matrix[key] for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def _materialize_cases(raw_cases: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return raw_cases or [{}]


def _format_template(template: str, insufficient_context_response: str) -> str:
    return template.replace("{insufficient_context_response}", insufficient_context_response)


def _build_resolved_spec(
    *,
    system: SystemConfig,
    registries: RegistryBundle,
    group_id: str,
    experiment_id: str,
    values: dict[str, Any],
) -> ResolvedSpec:
    required_keys = [
        "dataset_id",
        "evalset_id",
        "preprocessing_variant_id",
        "chunking_variant_id",
        "chunking_model_id",
        "store_backend_id",
        "store_index_variant_id",
        "store_embedding_model_id",
        "query_embedding_model_id",
        "search_algorithm_id",
        "reranker_id",
        "prompt_variant_id",
        "tool_policy_id",
        "generation_model_id",
    ]
    missing = [key for key in required_keys if key not in values]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Experiment is missing required axes: {missing_list}")

    dataset = registries.datasets[values["dataset_id"]]
    evalset = registries.evalsets[values["evalset_id"]]
    preprocessing = registries.preprocessing_variants[values["preprocessing_variant_id"]]
    chunking = registries.chunking_variants[values["chunking_variant_id"]]
    store_backend = registries.store_backends[values["store_backend_id"]]
    store_index = registries.store_index_variants[values["store_index_variant_id"]]
    store_embedding = registries.embedding_models[values["store_embedding_model_id"]]
    query_embedding = registries.embedding_models[values["query_embedding_model_id"]]
    search = registries.search_algorithms[values["search_algorithm_id"]]
    prompt = registries.prompt_variants[values["prompt_variant_id"]]
    if values["reranker_id"] not in registries.rerankers:
        raise ValueError(f"Unknown reranker_id: {values['reranker_id']}")
    if values["tool_policy_id"] not in registries.tool_policies:
        raise ValueError(f"Unknown tool_policy_id: {values['tool_policy_id']}")
    generation = registries.generation_models[values["generation_model_id"]]

    if store_index.backend_id != values["store_backend_id"]:
        raise ValueError(
            "Store index backend mismatch: "
            f"{values['store_index_variant_id']} belongs to {store_index.backend_id}, "
            f"not {values['store_backend_id']}"
        )

    load_variant_id = (
        f"load-{values['preprocessing_variant_id']}__"
        f"{values['chunking_variant_id']}__{values['chunking_model_id']}"
    )
    store_variant_id = (
        f"store-{values['store_backend_id']}__{values['store_index_variant_id']}__"
        f"{values['store_embedding_model_id']}"
    )
    inference_variant_id = (
        f"inf-{values['query_embedding_model_id']}__{values['search_algorithm_id']}__"
        f"{values['reranker_id']}__{values['prompt_variant_id']}__{values['tool_policy_id']}"
    )

    insufficient_context_response = system.generation.insufficient_context_response
    generation_temperature = float(
        values.get("generation_temperature", system.generation.default_temperature)
    )
    generation_max_tokens = int(values.get("generation_max_tokens", system.generation.default_max_tokens))

    return ResolvedSpec(
        group_id=group_id,
        experiment_id=experiment_id,
        dataset_id=values["dataset_id"],
        evalset_id=values["evalset_id"],
        preprocessing_variant_id=values["preprocessing_variant_id"],
        preprocessing_kind=preprocessing.kind,
        chunking_variant_id=values["chunking_variant_id"],
        chunking_kind=chunking.kind,
        chunking_model_id=values["chunking_model_id"],
        chunk_tokenizer=chunking.tokenizer,
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        store_backend_id=values["store_backend_id"],
        store_backend_kind=store_backend.kind,
        store_backend_runtime_status=store_backend.runtime_status,
        store_collection_prefix=store_backend.collection_prefix,
        store_index_variant_id=values["store_index_variant_id"],
        store_index_kind=store_index.kind,
        store_index_distance=store_index.distance,
        store_index_runtime_status=store_index.runtime_status,
        store_embedding_model_id=values["store_embedding_model_id"],
        store_embedding_provider=store_embedding.provider,
        store_embedding_model=store_embedding.model,
        store_embedding_batch_size=store_embedding.batch_size,
        store_embedding_pricing_input_per_1m_tokens_usd=(
            store_embedding.pricing_input_per_1m_tokens_usd
        ),
        query_embedding_model_id=values["query_embedding_model_id"],
        query_embedding_provider=query_embedding.provider,
        query_embedding_model=query_embedding.model,
        query_embedding_batch_size=query_embedding.batch_size,
        query_embedding_pricing_input_per_1m_tokens_usd=(
            query_embedding.pricing_input_per_1m_tokens_usd
        ),
        search_algorithm_id=values["search_algorithm_id"],
        search_kind=search.kind,
        top_k=search.top_k,
        mrr_depth=search.mrr_depth,
        reranker_id=values["reranker_id"],
        prompt_variant_id=values["prompt_variant_id"],
        prompt_system_prompt=_format_template(prompt.system_prompt.strip(), insufficient_context_response),
        prompt_user_template=prompt.user_template.strip(),
        tool_policy_id=values["tool_policy_id"],
        generation_model_id=values["generation_model_id"],
        generation_provider=generation.provider,
        generation_model=generation.model,
        generation_temperature=generation_temperature,
        generation_max_tokens=generation_max_tokens,
        generation_pricing_input_per_1m_tokens_usd=(
            generation.pricing_input_per_1m_tokens_usd
        ),
        generation_pricing_output_per_1m_tokens_usd=(
            generation.pricing_output_per_1m_tokens_usd
        ),
        insufficient_context_response=insufficient_context_response,
        price_catalog_version=system.pricing.price_catalog_version,
        runs_dir=system.paths.runs_dir,
        corpus_path=dataset.corpus_path,
        eval_path=evalset.eval_path,
        load_variant_id=load_variant_id,
        store_variant_id=store_variant_id,
        inference_variant_id=inference_variant_id,
        env=EnvironmentSettings(),
    )


def load_experiment_specs(
    experiment_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> list[ResolvedSpec]:
    system = _load_system_config()
    registries = _load_registry_bundle()
    group_data, overlay_data, _ = _load_experiment_files(experiment_path)

    combined = deep_merge(group_data, overlay_data)
    experiment_meta = combined.get("experiment", {})
    extends = combined.get("extends", {})

    fixed: dict[str, Any] = {}
    for family in ("load", "store", "inference"):
        prototype_id = extends.get(family)
        if prototype_id:
            fixed = deep_merge(fixed, _load_prototype(family, prototype_id).get("fixed", {}))

    fixed = deep_merge(fixed, group_data.get("fixed", {}))
    fixed = deep_merge(fixed, overlay_data.get("fixed", {}))

    matrix = deep_merge(group_data.get("matrix", {}), overlay_data.get("matrix", {}))
    points = _materialize_matrix(matrix)
    raw_cases = overlay_data.get("cases") or group_data.get("cases")
    cases = _materialize_cases(raw_cases)
    specs: list[ResolvedSpec] = []
    seen_keys: set[str] = set()
    for point in points:
        for case in cases:
            values = deep_merge(fixed, point)
            values = deep_merge(values, case)
            values = deep_merge(values, overrides or {})
            spec = _build_resolved_spec(
                system=system,
                registries=registries,
                group_id=experiment_meta["group_id"],
                experiment_id=experiment_meta["experiment_id"],
                values=values,
            )
            spec_key = spec.model_dump_json(exclude={"env"}, round_trip=True)
            if spec_key in seen_keys:
                continue
            seen_keys.add(spec_key)
            specs.append(spec)
    return specs
