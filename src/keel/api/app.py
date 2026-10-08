from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from keel import __version__
from keel.api.deps import Service, error_status
from keel.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    MetricsResponse,
    NodeView,
    RunRequest,
    RunResponse,
)
from keel.dag.graph import DAG, Node
from keel.errors import KeelError
from keel.scheduler.metrics import percentile


async def _demo_ingest(_ctx: dict[str, object]) -> object:
    return {"records": 128}


async def _demo_summarise(ctx: dict[str, object]) -> object:
    return {"summary": "ok", "from": ctx.get("_demo_ingest")}


DEMO_DAG = DAG(
    nodes=(
        Node(name="ingest", run=_demo_ingest),
        Node(name="summarise", run=_demo_summarise, depends_on=("ingest",)),
    )
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = Service.build()
    service.register_dag("demo", DEMO_DAG)
    app.state.service = service
    try:
        yield
    finally:
        await service.aclose()


def create_app(service: Service | None = None) -> FastAPI:
    app = FastAPI(title="keel", version=__version__, lifespan=lifespan)

    if service is not None:
        app.state.service = service

    def current(request: Request) -> Service:
        found: Service = request.app.state.service
        return found

    @app.exception_handler(KeelError)
    async def handle_keel_error(_request: Request, exc: KeelError) -> JSONResponse:
        return JSONResponse(status_code=error_status(exc), content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        svc = current(request)
        return HealthResponse(
            status="ok",
            device=svc.engine.device.name,
            kv_blocks=svc.engine.kv.pool.num_blocks,
            context_length=svc.engine.context_length,
            free_blocks=svc.engine.kv.num_free_blocks,
            version=__version__,
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(
        request: Request, body: CompletionRequest
    ) -> CompletionResponse | JSONResponse:
        svc = current(request)
        estimate = body.max_tokens * 10
        wait_ms = svc.admit(body.tenant_id, estimate)
        if wait_ms > 0:
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limited", "retry_after_ms": round(wait_ms, 2)},
                headers={"Retry-After": str(max(1, round(wait_ms / 1000.0)))},
            )

        result = await svc.complete(
            body.prompt,
            model=body.model,
            max_tokens=body.max_tokens,
            stop=body.stop,
            temperature=body.temperature,
            tenant_id=body.tenant_id,
            allow_semantic_cache=body.allow_semantic_cache,
        )
        svc.budgets.settle(body.tenant_id, estimate, result.cost_micros)
        return CompletionResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            cache_tier=result.cache_tier,
            cache_similarity=result.cache_similarity,
            cost_micros=result.cost_micros,
        )

    @app.get("/v1/metrics", response_model=MetricsResponse)
    async def metrics(request: Request) -> MetricsResponse:
        svc = current(request)
        cache = svc.cache.stats
        remaining = min(
            (svc.budgets.remaining_micros(t) for t in svc.tenants),
            default=svc.settings.policy.default_budget_micros,
        )
        return MetricsResponse(
            requests=svc.usage.requests,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            cache_hit_rate=cache.hit_rate,
            prompt_tokens=svc.usage.prompt_tokens,
            completion_tokens=svc.usage.completion_tokens,
            cost_micros=svc.usage.cost_micros,
            ttft_p50_ms=percentile(svc.usage.ttfts, 0.5),
            ttft_p95_ms=percentile(svc.usage.ttfts, 0.95),
            remaining_budget_micros=remaining,
        )

    @app.post("/v1/runs", response_model=RunResponse)
    async def start_run(request: Request, body: RunRequest) -> RunResponse | JSONResponse:
        svc = current(request)
        if body.dag not in svc.dags:
            return JSONResponse(status_code=404, content={"detail": f"unknown dag {body.dag}"})
        run_id, outcome, reused = await svc.run_dag(body.dag, body.partition, body.inputs)
        return RunResponse(
            run_id=run_id,
            dag=body.dag,
            status="succeeded" if outcome.ok else "failed",
            nodes=[
                NodeView(
                    name=name,
                    # A reused node returns a value, so the executor sees a
                    # success. Reporting CACHED is what tells an operator the
                    # work did not actually happen.
                    status="cached" if name in reused else result.status.value,
                    attempts=result.attempts,
                    error=result.error,
                )
                for name, result in outcome.results.items()
            ],
        )

    @app.get("/v1/traces", response_model=list[dict[str, object]])
    async def traces(request: Request) -> list[dict[str, object]]:
        return current(request).tracer.as_dicts()

    return app


app = create_app()
