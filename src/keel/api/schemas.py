from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "sim-7b"
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: tuple[str, ...] = ()
    tenant_id: str = "default"
    deadline_ms: float | None = None
    allow_semantic_cache: bool = True


class CompletionResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    ttft_ms: float
    tpot_ms: float | None
    cache_tier: str
    cache_similarity: float | None = None
    cost_micros: int


class MetricsResponse(BaseModel):
    requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    prompt_tokens: int
    completion_tokens: int
    cost_micros: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    remaining_budget_micros: int


class HealthResponse(BaseModel):
    status: str
    device: str
    kv_blocks: int
    context_length: int
    free_blocks: int
    version: str


class RunRequest(BaseModel):
    dag: str
    partition: str | None = None
    inputs: dict[str, object] = Field(default_factory=dict)


class NodeView(BaseModel):
    name: str
    status: str
    attempts: int
    error: str | None = None


class RunResponse(BaseModel):
    run_id: str
    dag: str
    status: str
    nodes: list[NodeView]


class ErrorResponse(BaseModel):
    detail: str
    retry_after_ms: int | None = None
