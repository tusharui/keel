from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from keel.api.app import create_app
from keel.api.deps import Service
from keel.config import CacheConfig, EngineConfig, PolicyConfig, Settings
from keel.db.base import Base


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        engine=EngineConfig(kv_blocks=512, block_size=16, context_length=2048),
        cache=CacheConfig(exact_max_entries=64, semantic_enabled=False),
        policy=PolicyConfig(rate_limit_per_s=5.0, rate_limit_burst=5.0),
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    service = Service.build(settings)
    service.register_dag("demo", _dag())

    # The API expects migrations to have run, as it would in production. Creating
    # the schema here keeps the test to the handler rather than the migration
    # runner, and the migration suite covers that separately.
    async with service.db_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    app = create_app(service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    await service.aclose()


def _dag():
    from keel.dag import DAG, Node

    async def a(_ctx: dict[str, object]) -> object:
        return "a"

    async def b(ctx: dict[str, object]) -> object:
        return f"b<-{ctx.get('a')}"

    return DAG(nodes=(Node(name="a", run=a), Node(name="b", run=b, depends_on=("a",))))


# --- health ---------------------------------------------------------------------------


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["kv_blocks"] == 512
    assert body["free_blocks"] == 512


# --- completions ---------------------------------------------------------------------


async def test_completion_returns_generated_text(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/completions", json={"prompt": "what is the capital of france", "max_tokens": 16}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"]
    assert body["completion_tokens"] >= 0
    assert body["cache_tier"] == "miss"
    assert body["ttft_ms"] > 0
    assert body["cost_micros"] > 0


async def test_repeated_prompt_hits_the_exact_cache(client: httpx.AsyncClient) -> None:
    payload = {"prompt": "summarise the revenue report", "max_tokens": 16}
    first = await client.post("/v1/completions", json=payload)
    second = await client.post("/v1/completions", json=payload)

    assert first.json()["cache_tier"] == "miss"
    assert second.json()["cache_tier"] == "exact"
    assert second.json()["text"] == first.json()["text"]
    assert second.json()["cost_micros"] == 0


async def test_empty_prompt_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/completions", json={"prompt": ""})
    assert response.status_code == 422


async def test_max_tokens_bounds_are_enforced(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 99999})
    assert response.status_code == 422


async def test_rate_limit_returns_429_with_retry_after(
    client: httpx.AsyncClient,
) -> None:
    payload = {"prompt": "hello there", "max_tokens": 8}
    statuses = [(await client.post("/v1/completions", json=payload)).status_code for _ in range(8)]
    assert 429 in statuses
    assert statuses[0] == 200


async def test_429_carries_a_retry_after_header(client: httpx.AsyncClient) -> None:
    payload = {"prompt": "burst test", "max_tokens": 8}
    for _ in range(5):
        await client.post("/v1/completions", json=payload)
    response = await client.post("/v1/completions", json=payload)
    if response.status_code == 429:
        assert "retry_after_ms" in response.json()
        assert int(response.headers["Retry-After"]) >= 1


async def test_budget_exhaustion_returns_402(client: httpx.AsyncClient) -> None:
    service: Service = client._transport.app.state.service  # type: ignore[attr-defined]
    service.seed_budget("broke", 1)

    response = await client.post(
        "/v1/completions",
        json={"prompt": "hello", "max_tokens": 64, "tenant_id": "broke"},
    )
    assert response.status_code == 402
    assert "limit" in response.json()["detail"]


# --- metrics -------------------------------------------------------------------------


async def test_metrics_reflect_traffic(client: httpx.AsyncClient) -> None:
    await client.post("/v1/completions", json={"prompt": "first", "max_tokens": 8})
    await client.post("/v1/completions", json={"prompt": "first", "max_tokens": 8})

    body = (await client.get("/v1/metrics")).json()
    assert body["requests"] == 2
    assert body["cache_hits"] == 1
    assert body["cache_misses"] == 1
    assert body["cache_hit_rate"] == pytest.approx(0.5)
    assert body["completion_tokens"] >= 0


async def test_metrics_on_a_fresh_service(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/metrics")).json()
    assert body["requests"] == 0
    assert body["cache_hit_rate"] == 0.0


# --- dag runs ------------------------------------------------------------------------


async def test_run_executes_the_dag(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/runs", json={"dag": "demo", "partition": "p1"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert {n["name"] for n in body["nodes"]} == {"a", "b"}
    assert all(n["status"] == "succeeded" for n in body["nodes"])


async def test_unknown_dag_is_404(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/runs", json={"dag": "nope"})
    assert response.status_code == 404
    assert "unknown dag" in response.json()["detail"]


async def test_404_is_documented_in_the_schema(client: httpx.AsyncClient) -> None:
    """A hand-built JSONResponse is invisible to the docs, which is how a
    documented endpoint ends up with an undocumented failure mode."""
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert "404" in paths["/v1/runs"]["post"]["responses"]


async def test_run_request_example_uses_a_registered_dag(client: httpx.AsyncClient) -> None:
    """The example in the docs should not 404 for anyone following them."""
    schema = (await client.get("/openapi.json")).json()["components"]["schemas"]
    assert schema["RunRequest"]["properties"]["dag"]["default"] == "demo"


async def test_rerun_reuses_persisted_nodes(client: httpx.AsyncClient) -> None:
    payload = {"dag": "demo", "partition": "p2"}
    first = (await client.post("/v1/runs", json=payload)).json()
    second = (await client.post("/v1/runs", json=payload)).json()

    assert first["run_id"] == second["run_id"], "a rerun resumes the same run"
    statuses = {n["name"]: n["status"] for n in second["nodes"]}
    assert statuses == {"a": "cached", "b": "cached"}


# --- traces ---------------------------------------------------------------------------


async def test_traces_are_recorded(client: httpx.AsyncClient) -> None:
    await client.post("/v1/completions", json={"prompt": "trace me", "max_tokens": 8})
    spans = (await client.get("/v1/traces")).json()
    names = {s["name"] for s in spans}
    assert {"completion", "generate"} <= names

    generate = next(s for s in spans if s["name"] == "generate")
    completion = next(s for s in spans if s["name"] == "completion")
    assert generate["parent_id"] == completion["span_id"]


async def test_trace_durations_are_measured(client: httpx.AsyncClient) -> None:
    """The service never sleeps for simulated latency, so a tracer sharing the
    simulation clock would record every span as 0ms."""
    await client.post("/v1/completions", json={"prompt": "time me", "max_tokens": 8})
    spans = (await client.get("/v1/traces")).json()
    assert all(s["duration_ms"] > 0 for s in spans)


async def test_trace_carries_the_simulated_figures(client: httpx.AsyncClient) -> None:
    """Wall-clock says how long the simulation took; the simulated numbers say
    what a real backend would have cost, which is the useful one here."""
    await client.post("/v1/completions", json={"prompt": "sim me", "max_tokens": 8})
    spans = (await client.get("/v1/traces")).json()
    generate = next(s for s in spans if s["name"] == "generate")
    assert generate["attributes"]["sim_ttft_ms"] > 0
    assert generate["attributes"]["sim_duration_ms"] > 0


async def test_deadline_is_not_a_completion_field(client: httpx.AsyncClient) -> None:
    """It is scheduler state. Accepting it here and ignoring it would be a field
    that looks like a feature and does nothing."""
    properties = (await client.get("/openapi.json")).json()["components"]["schemas"][
        "CompletionRequest"
    ]["properties"]
    assert "deadline_ms" not in properties


# --- schema ---------------------------------------------------------------------------


async def test_openapi_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/completions" in response.json()["paths"]
