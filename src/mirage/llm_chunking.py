from __future__ import annotations

from collections.abc import Callable, Iterable
import hashlib
import json
import math
from pathlib import Path
import re
from time import sleep

import httpx
from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
import tiktoken

from mirage.config import ChunkingConfig, EnvironmentSettings
from mirage.schemas import Chunk, Document


class SourceUnit(BaseModel):
    unit_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    unit_type: str = "paragraph"


class BoundaryDecision(BaseModel):
    boundary_unit_id: str | None = None
    reason: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)


class BatchBoundaryDecision(BoundaryDecision):
    request_id: str


class BatchBoundaryResponse(BaseModel):
    decisions: list[BatchBoundaryDecision]

    @field_validator("decisions")
    @classmethod
    def _decisions_not_empty(cls, decisions: list[BatchBoundaryDecision]) -> list[BatchBoundaryDecision]:
        if not decisions:
            raise ValueError("batch response must contain decisions")
        return decisions


class LlmChunkPlan(BaseModel):
    document_hash: str
    model: str
    prompt_version: str
    schema_version: str
    chunks: list[list[str]]
    decisions: list[BoundaryDecision]

    @field_validator("chunks")
    @classmethod
    def _chunks_not_empty(cls, chunks: list[list[str]]) -> list[list[str]]:
        if not chunks:
            raise ValueError("chunk plan must contain at least one chunk")
        if any(not chunk for chunk in chunks):
            raise ValueError("chunk plan contains an empty chunk")
        return chunks


class LlmChunkingPreflight(BaseModel):
    documents: int
    units: int
    tokens: int
    cached_plans: int
    missing_plans: int
    estimated_llm_calls: int
    estimated_llm_batch_calls: int
    model: str


class OpenRouterBoundaryPlanner:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        env: EnvironmentSettings,
        max_retries: int,
        temperature: float = 0.0,
        rate_limit_backoff_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        if provider != "openrouter":
            raise NotImplementedError(f"Unsupported LLM chunking provider: {provider}")
        if not env.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for LLM semantic chunking")
        http_client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0))
        self._client = OpenAI(
            api_key=env.openrouter_api_key,
            base_url=env.openrouter_base_url,
            http_client=http_client,
            max_retries=0,
        )
        self._model = model
        self._max_retries = max_retries
        self._temperature = temperature
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._sleep = sleep_fn

    def decide_boundary(self, *, units: list[SourceUnit], max_chunk_tokens: int) -> BoundaryDecision:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document segmentation model for RAG ingestion. "
                    "Read ordered source units and choose the first unit where a new semantic chunk should start. "
                    "Return strict JSON only. Do not rewrite source text. "
                    "Use null boundary_unit_id only if all provided units belong in one coherent chunk."
                ),
            },
            {"role": "user", "content": self._prompt(units, max_chunk_tokens, None)},
        ]
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                messages.append({"role": "user", "content": self._prompt(units, max_chunk_tokens, last_error)})
            response = self._create_completion(messages)
            raw = response.choices[0].message.content or ""
            try:
                decision = BoundaryDecision.model_validate(json.loads(raw))
                self._validate_decision(decision, units, max_chunk_tokens)
                return decision
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                last_error = str(error)
                messages.append({"role": "assistant", "content": raw})
        raise ValueError(f"LLM chunk boundary output stayed invalid after retries: {last_error}")

    def decide_boundaries(
        self,
        requests: list[tuple[str, list[SourceUnit], int]],
    ) -> dict[str, BoundaryDecision]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document segmentation model for RAG ingestion. "
                    "For each request, read ordered source units and choose the first unit where a new semantic chunk should start. "
                    "Return strict JSON only with one decision per request_id. Do not rewrite source text. "
                    "Use null boundary_unit_id only if all units in that request belong in one coherent chunk."
                ),
            },
            {"role": "user", "content": self._batch_prompt(requests, None)},
        ]
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                messages.append({"role": "user", "content": self._batch_prompt(requests, last_error)})
            response = self._create_completion(messages)
            raw = response.choices[0].message.content or ""
            try:
                parsed = BatchBoundaryResponse.model_validate(json.loads(raw))
                decisions = {decision.request_id: BoundaryDecision.model_validate(decision.model_dump()) for decision in parsed.decisions}
                expected_ids = {request_id for request_id, _, _ in requests}
                if set(decisions) != expected_ids:
                    raise ValueError("batch response must contain exactly one decision per request_id")
                for request_id, units, max_chunk_tokens in requests:
                    self._validate_decision(decisions[request_id], units, max_chunk_tokens)
                return decisions
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                last_error = str(error)
                messages.append({"role": "assistant", "content": raw})
        raise ValueError(f"LLM batch chunk boundary output stayed invalid after retries: {last_error}")

    def _create_completion(self, messages: list[dict[str, str]]) -> object:
        while True:
            try:
                return self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except APIStatusError as error:
                if error.status_code == 429 or error.status_code >= 500:
                    self._sleep(self._rate_limit_backoff_seconds)
                    continue
                raise

    def _prompt(self, units: list[SourceUnit], max_chunk_tokens: int, error: str | None) -> str:
        payload = {
            "max_chunk_tokens": max_chunk_tokens,
            "output_schema": {
                "boundary_unit_id": "string or null; first unit_id that starts the next chunk",
                "reason": "short reason for the boundary decision",
                "confidence": "number between 0 and 1",
            },
            "units": [unit.model_dump(mode="json") for unit in units],
        }
        if error:
            payload["previous_validation_error"] = error
        return json.dumps(payload, ensure_ascii=False)

    def _batch_prompt(self, requests: list[tuple[str, list[SourceUnit], int]], error: str | None) -> str:
        payload = {
            "output_schema": {
                "decisions": [
                    {
                        "request_id": "same request_id as input",
                        "boundary_unit_id": "string or null; first unit_id that starts the next chunk for this request",
                        "reason": "short reason for the boundary decision",
                        "confidence": "number between 0 and 1",
                    }
                ]
            },
            "requests": [
                {
                    "request_id": request_id,
                    "max_chunk_tokens": max_chunk_tokens,
                    "units": [unit.model_dump(mode="json") for unit in units],
                }
                for request_id, units, max_chunk_tokens in requests
            ],
        }
        if error:
            payload["previous_validation_error"] = error
        return json.dumps(payload, ensure_ascii=False)

    def _validate_decision(self, decision: BoundaryDecision, units: list[SourceUnit], max_chunk_tokens: int) -> None:
        validate_boundary_decision(decision, units, max_chunk_tokens)


class LlmSemanticChunker:
    def __init__(
        self,
        config: ChunkingConfig,
        *,
        boundary_planner: Callable[[list[SourceUnit], int], BoundaryDecision],
        batch_boundary_planner: Callable[[list[tuple[str, list[SourceUnit], int]]], dict[str, BoundaryDecision]] | None = None,
        batch_size: int = 16,
        cache_dir: Path | None = None,
        model: str = "test-model",
        prompt_version: str = "llm-semantic-boundary-v1",
        schema_version: str = "llm-chunk-plan-v1",
    ) -> None:
        self._config = config
        self._encoding = tiktoken.get_encoding(config.tokenizer)
        self._planner = boundary_planner
        self._batch_planner = batch_boundary_planner
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._model = model
        self._prompt_version = prompt_version
        self._schema_version = schema_version

    def unitize_document(self, document: Document) -> list[SourceUnit]:
        units: list[SourceUnit] = []
        position = 0
        order = 0
        parts = _split_paragraphs(document.text)
        for text in parts:
            start = document.text.find(text, position)
            if start < 0:
                start = position
            end = start + len(text)
            position = end
            token_count = len(self._encoding.encode(text))
            if token_count == 0:
                continue
            units.append(
                SourceUnit(
                    unit_id=f"u_{order:04d}",
                    text=text,
                    start_char=start,
                    end_char=end,
                    token_count=token_count,
                    unit_type=_unit_type(text),
                )
            )
            order += 1
        return units

    def preflight(self, documents: Iterable[Document]) -> LlmChunkingPreflight:
        docs = list(documents)
        units = 0
        tokens = 0
        cached = 0
        for document in docs:
            doc_units = self.unitize_document(document)
            units += len(doc_units)
            tokens += sum(unit.token_count for unit in doc_units)
            if self._cache_path(document).exists():
                cached += 1
        missing = len(docs) - cached
        return LlmChunkingPreflight(
            documents=len(docs),
            units=units,
            tokens=tokens,
            cached_plans=cached,
            missing_plans=missing,
            estimated_llm_calls=missing,
            estimated_llm_batch_calls=math.ceil(missing / self._batch_size) if self._batch_planner is not None else missing,
            model=self._model,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        units = self.unitize_document(document)
        if not units:
            return []
        plan = self._load_plan(document, units)
        if plan is None:
            plan = self._build_plan(document, units)
            self._write_plan(document, plan)
        return self._materialize(document, units, plan)

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        docs = list(documents)
        if self._batch_planner is not None:
            return self._chunk_documents_batched(docs)
        output: list[Chunk] = []
        for document in docs:
            output.extend(self.chunk_document(document))
        return output

    def _chunk_documents_batched(self, documents: list[Document]) -> list[Chunk]:
        outputs: list[list[Chunk]] = [[] for _ in documents]
        states: list[dict[str, object]] = []
        for index, document in enumerate(documents):
            units = self.unitize_document(document)
            if not units:
                continue
            plan = self._load_plan(document, units)
            if plan is not None:
                outputs[index] = self._materialize(document, units, plan)
                continue
            states.append({"index": index, "document": document, "units": units, "chunks": [], "decisions": [], "start": 0})

        active = states
        while active:
            requests: list[tuple[str, list[SourceUnit], int]] = []
            request_states: dict[str, dict[str, object]] = {}
            next_active: list[dict[str, object]] = []
            finished_ids: set[int] = set()
            for state in active:
                units = state["units"]
                start = state["start"]
                if not isinstance(units, list) or not isinstance(start, int):
                    raise TypeError("Invalid LLM chunking state")
                window = self._window(units, start)
                if start > 0 and start + len(window) >= len(units) and sum(unit.token_count for unit in window) <= self._config.chunk_size:
                    state_chunks = state["chunks"]
                    if not isinstance(state_chunks, list):
                        raise TypeError("Invalid LLM chunking chunks state")
                    state_chunks.append([unit.unit_id for unit in window])
                    self._finish_batched_state(state, outputs)
                    finished_ids.add(id(state))
                    continue
                request_id = f"r_{len(requests):04d}"
                requests.append((request_id, window, self._config.chunk_size))
                request_states[request_id] = state
                if len(requests) >= self._batch_size:
                    break

            if requests:
                if self._batch_planner is None:
                    raise RuntimeError("Batch planner is not configured")
                decisions = self._batch_planner(requests)
                for request_id, decision in decisions.items():
                    state = request_states[request_id]
                    units = state["units"]
                    start = state["start"]
                    if not isinstance(units, list) or not isinstance(start, int):
                        raise TypeError("Invalid LLM chunking state")
                    window = self._window(units, start)
                    validate_boundary_decision(decision, window, self._config.chunk_size)
                    state_decisions = state["decisions"]
                    state_chunks = state["chunks"]
                    if not isinstance(state_decisions, list) or not isinstance(state_chunks, list):
                        raise TypeError("Invalid LLM chunking decision state")
                    state_decisions.append(decision)
                    if decision.boundary_unit_id is None:
                        boundary_offset = len(window)
                    else:
                        boundary_offset = next(index for index, unit in enumerate(window) if unit.unit_id == decision.boundary_unit_id)
                    if boundary_offset <= 0:
                        raise ValueError("LLM chunk boundary must not point at the first unit")
                    state_chunks.append([unit.unit_id for unit in units[start : start + boundary_offset]])
                    state["start"] = start + boundary_offset
                    if state["start"] >= len(units):
                        self._finish_batched_state(state, outputs)
                        finished_ids.add(id(state))
                    else:
                        next_active.append(state)

            processed_ids = {id(state) for state in request_states.values()}
            for state in active:
                if id(state) not in processed_ids and id(state) not in finished_ids and state not in next_active:
                    next_active.append(state)
            active = next_active

        return [chunk for group in outputs for chunk in group]

    def _finish_batched_state(self, state: dict[str, object], outputs: list[list[Chunk]]) -> None:
        document = state["document"]
        units = state["units"]
        if not isinstance(document, Document) or not isinstance(units, list):
            raise TypeError("Invalid completed LLM chunking state")
        plan = LlmChunkPlan(
            document_hash=_document_hash(document),
            model=self._model,
            prompt_version=self._prompt_version,
            schema_version=self._schema_version,
            chunks=state["chunks"],
            decisions=state["decisions"],
        )
        validate_chunk_plan(plan, units)
        self._write_plan(document, plan)
        index = state["index"]
        if not isinstance(index, int):
            raise TypeError("Invalid completed LLM chunking index")
        outputs[index] = self._materialize(document, units, plan)

    def _build_plan(self, document: Document, units: list[SourceUnit]) -> LlmChunkPlan:
        chunks: list[list[str]] = []
        decisions: list[BoundaryDecision] = []
        start = 0
        while start < len(units):
            window = self._window(units, start)
            if start > 0 and start + len(window) >= len(units) and sum(unit.token_count for unit in window) <= self._config.chunk_size:
                chunks.append([unit.unit_id for unit in window])
                break
            decision = self._planner(window, self._config.chunk_size)
            validate_boundary_decision(decision, window, self._config.chunk_size)
            decisions.append(decision)
            if decision.boundary_unit_id is None:
                boundary_offset = len(window)
            else:
                boundary_offset = next(index for index, unit in enumerate(window) if unit.unit_id == decision.boundary_unit_id)
            if boundary_offset <= 0:
                raise ValueError("LLM chunk boundary must not point at the first unit")
            chunks.append([unit.unit_id for unit in units[start : start + boundary_offset]])
            start += boundary_offset
        plan = LlmChunkPlan(
            document_hash=_document_hash(document),
            model=self._model,
            prompt_version=self._prompt_version,
            schema_version=self._schema_version,
            chunks=chunks,
            decisions=decisions,
        )
        validate_chunk_plan(plan, units)
        return plan

    def _window(self, units: list[SourceUnit], start: int) -> list[SourceUnit]:
        output: list[SourceUnit] = []
        token_count = 0
        for unit in units[start:]:
            if output and token_count + unit.token_count > self._config.chunk_size * 2:
                break
            output.append(unit)
            token_count += unit.token_count
            if token_count >= self._config.chunk_size:
                break
        return output

    def _materialize(self, document: Document, units: list[SourceUnit], plan: LlmChunkPlan) -> list[Chunk]:
        unit_by_id = {unit.unit_id: unit for unit in units}
        chunks: list[Chunk] = []
        for order, unit_ids in enumerate(plan.chunks):
            parts = [unit_by_id[unit_id].text for unit_id in unit_ids]
            text = "\n\n".join(parts).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::chunk-{order:04d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    source=document.source,
                    text=text,
                    order=order,
                    token_count=len(self._encoding.encode(text)),
                    metadata={**document.metadata, "chunking_strategy": "llm-semantic", "unit_ids": unit_ids},
                )
            )
        return chunks

    def _cache_path(self, document: Document) -> Path:
        if self._cache_dir is None:
            return Path("")
        doc_hash = _document_hash(document)
        return self._cache_dir / self._model.replace("/", "-").replace(":", "-") / f"{doc_hash}.json"

    def _load_plan(self, document: Document, units: list[SourceUnit]) -> LlmChunkPlan | None:
        if self._cache_dir is None:
            return None
        path = self._cache_path(document)
        if not path.exists():
            return None
        plan = LlmChunkPlan.model_validate(json.loads(path.read_text()))
        validate_chunk_plan(plan, units)
        return plan

    def _write_plan(self, document: Document, plan: LlmChunkPlan) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_path(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n")


def validate_boundary_decision(decision: BoundaryDecision, units: list[SourceUnit], max_chunk_tokens: int) -> None:
    ids = [unit.unit_id for unit in units]
    if decision.boundary_unit_id is not None and decision.boundary_unit_id not in ids:
        raise ValueError(f"Unknown boundary_unit_id: {decision.boundary_unit_id}")
    if decision.boundary_unit_id == ids[0]:
        raise ValueError("boundary_unit_id must not be the first unit")
    if decision.boundary_unit_id is None and sum(unit.token_count for unit in units) > max_chunk_tokens:
        raise ValueError("boundary_unit_id is required when the candidate window exceeds max_chunk_tokens")


def validate_chunk_plan(plan: LlmChunkPlan, units: list[SourceUnit]) -> None:
    expected_ids = [unit.unit_id for unit in units]
    planned_ids = [unit_id for chunk in plan.chunks for unit_id in chunk]
    if planned_ids != expected_ids:
        raise ValueError("LLM chunk plan must cover all units exactly once in order")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_paragraphs(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", stripped) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()] or [stripped]


def _unit_type(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("#"):
        return "heading"
    if stripped.startswith(('-', '*', '+')):
        return "list"
    return "paragraph"


def _document_hash(document: Document) -> str:
    payload = json.dumps(
        {"doc_id": document.doc_id, "title": document.title, "source": document.source, "text": document.text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
