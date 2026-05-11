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
    semantic_similarity_threshold: float = 0.8
    semantic_min_sentences_per_chunk: int = 2


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qdrant_url: str = "http://localhost:6333"


class SystemPaths(BaseModel):
    artifacts_dir: str = "artifacts"


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
    adapter_id: str = "jsonl"
    dataset_root: str | None = None
    corpus_path: str | None = None


class EvalSetEntry(BaseModel):
    adapter_id: str = "jsonl"
    eval_root: str | None = None
    eval_path: str | None = None
    split: str = "dev"


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
    dense_top_k: int | None = None
    sparse_top_k: int | None = None
    rrf_k: int = 60
    mmr_lambda: float = 0.5


class PromptVariantEntry(BaseModel):
    system_prompt: str
    user_template: str


class RerankerEntry(BaseModel):
    kind: str
    model: str | None = None
    batch_size: int = 64


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
    semantic_similarity_threshold: float = 0.8
    semantic_min_sentences_per_chunk: int = 2
    semantic_embedding_provider: str = "none"
    semantic_embedding_model: str = "none"
    semantic_embedding_batch_size: int = 1
    semantic_embedding_pricing_input_per_1m_tokens_usd: float = 0.0
    store_backend_id: str
    store_backend_kind: str
    store_backend_runtime_status: str
    store_collection_prefix: str
    store_index_variant_id: str
    store_index_kind: str
    store_index_distance: str
    store_index_runtime_status: str
    store_index_nlist: int | None = None
    store_index_m: int | None = None
    store_index_bits: int | None = None
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
    search_dense_top_k: int | None = None
    search_sparse_top_k: int | None = None
    search_rrf_k: int = 60
    search_mmr_lambda: float = 0.5
    reranker_id: str
    reranker_kind: str
    reranker_model: str | None = None
    reranker_batch_size: int = 64
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
    artifacts_dir: str
    dataset_adapter_id: str
    dataset_root: str | None = None
    eval_adapter_id: str
    eval_root: str | None = None
    eval_split: str = "dev"
    corpus_path: str | None = None
    eval_path: str | None = None
    load_variant_id: str
    store_variant_id: str
    inference_variant_id: str
    env: EnvironmentSettings = Field(default_factory=EnvironmentSettings)


def _load_system_config() -> SystemConfig:
    raw = load_toml_file(PROJECT_ROOT / "config" / "defaults.toml")
    return SystemConfig.model_validate(raw)


def _load_registry_bundle() -> RegistryBundle:
    raw = load_toml_dir(PROJECT_ROOT / "config")
    return RegistryBundle.model_validate(raw)


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
    reranker = registries.rerankers[values["reranker_id"]]
    if values["tool_policy_id"] not in registries.tool_policies:
        raise ValueError(f"Unknown tool_policy_id: {values['tool_policy_id']}")
    generation = registries.generation_models[values["generation_model_id"]]
    semantic_embedding = registries.embedding_models.get(values["chunking_model_id"])

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
        semantic_similarity_threshold=float(values.get("semantic_similarity_threshold", 0.8)),
        semantic_min_sentences_per_chunk=int(values.get("semantic_min_sentences_per_chunk", 2)),
        semantic_embedding_provider=(semantic_embedding.provider if semantic_embedding is not None else "none"),
        semantic_embedding_model=(semantic_embedding.model if semantic_embedding is not None else "none"),
        semantic_embedding_batch_size=(semantic_embedding.batch_size if semantic_embedding is not None else 1),
        semantic_embedding_pricing_input_per_1m_tokens_usd=(
            semantic_embedding.pricing_input_per_1m_tokens_usd if semantic_embedding is not None else 0.0
        ),
        store_backend_id=values["store_backend_id"],
        store_backend_kind=store_backend.kind,
        store_backend_runtime_status=store_backend.runtime_status,
        store_collection_prefix=store_backend.collection_prefix,
        store_index_variant_id=values["store_index_variant_id"],
        store_index_kind=store_index.kind,
        store_index_distance=store_index.distance,
        store_index_runtime_status=store_index.runtime_status,
        store_index_nlist=getattr(store_index, "nlist", None),
        store_index_m=getattr(store_index, "m", None),
        store_index_bits=getattr(store_index, "bits", None),
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
        search_dense_top_k=search.dense_top_k,
        search_sparse_top_k=search.sparse_top_k,
        search_rrf_k=search.rrf_k,
        search_mmr_lambda=search.mmr_lambda,
        reranker_id=values["reranker_id"],
        reranker_kind=reranker.kind,
        reranker_model=reranker.model,
        reranker_batch_size=reranker.batch_size,
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
        artifacts_dir=system.paths.artifacts_dir,
        dataset_adapter_id=dataset.adapter_id,
        dataset_root=dataset.dataset_root,
        eval_adapter_id=evalset.adapter_id,
        eval_root=evalset.eval_root,
        eval_split=evalset.split,
        corpus_path=dataset.corpus_path,
        eval_path=evalset.eval_path,
        load_variant_id=load_variant_id,
        store_variant_id=store_variant_id,
        inference_variant_id=inference_variant_id,
        env=EnvironmentSettings(),
    )


def _study_dir(experiment_path: str | Path) -> Path:
    path = PROJECT_ROOT / Path(experiment_path)
    if path.is_file():
        return path.parent
    return path


def _study_fixed(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key not in {"id", "matrix", "cases"}}


def _selected_study_experiment(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = overrides.pop("study_experiment_id", None)
    if selected_id is None or selected_id == "baseline":
        return None
    for experiment in raw.get("experiments", []):
        if experiment.get("id") == selected_id:
            return experiment
    raise ValueError(f"Unknown study_experiment_id: {selected_id}")


def _load_study_specs(
    experiment_path: str | Path,
    *,
    system: SystemConfig,
    registries: RegistryBundle,
    overrides: dict[str, Any],
) -> list[ResolvedSpec]:
    raw = load_toml_file(_study_dir(experiment_path) / "study.toml")
    study = raw["study"]
    baseline = raw["baseline"]
    selected = _selected_study_experiment(raw, overrides)
    fixed = _study_fixed(baseline)
    matrix = baseline.get("matrix", {})
    cases = baseline.get("cases")
    experiment_id = baseline.get("id", "baseline")
    if selected is not None:
        fixed = deep_merge(fixed, _study_fixed(selected))
        matrix = deep_merge(matrix, selected.get("matrix", {}))
        cases = selected.get("cases") or cases
        experiment_id = selected["id"]
    specs: list[ResolvedSpec] = []
    seen_keys: set[str] = set()
    for point in _materialize_matrix(matrix):
        for case in _materialize_cases(cases):
            values = deep_merge(fixed, point)
            values = deep_merge(values, case)
            values = deep_merge(values, overrides)
            spec = _build_resolved_spec(
                system=system,
                registries=registries,
                group_id=study["id"],
                experiment_id=experiment_id,
                values=values,
            )
            spec_key = spec.model_dump_json(exclude={"env"}, round_trip=True)
            if spec_key in seen_keys:
                continue
            seen_keys.add(spec_key)
            specs.append(spec)
    return specs


def load_experiment_specs(
    experiment_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> list[ResolvedSpec]:
    system = _load_system_config()
    registries = _load_registry_bundle()
    parsed_overrides = dict(overrides or {})
    study_dir = _study_dir(experiment_path)
    if (study_dir / "study.toml").exists():
        return _load_study_specs(
            experiment_path,
            system=system,
            registries=registries,
            overrides=parsed_overrides,
        )
    raise ValueError(f"Study file not found under: {study_dir}")
