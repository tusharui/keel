from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from keel.db.models import Budget, CacheEntry, InferenceRequest, Node, Run, RunEvent, Tenant
from keel.enums import CacheTier, NodeStatus, RequestStatus


async def test_tenant_and_request_round_trip(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.commit()

    request = InferenceRequest(
        tenant_id=tenant.id,
        model="sim-7b",
        prompt_hash="a" * 64,
        max_tokens=128,
        prompt_tokens=40,
        completion_tokens=96,
        cost_micros=1_500,
        status=RequestStatus.SUCCEEDED,
        cache_tier=CacheTier.PREFIX,
        ttft_ms=82.5,
        tpot_ms=11.2,
    )
    session.add(request)
    await session.commit()

    loaded = await session.get(InferenceRequest, request.id)
    assert loaded is not None
    assert loaded.status is RequestStatus.SUCCEEDED
    assert loaded.cache_tier is CacheTier.PREFIX
    assert loaded.cost_micros == 1_500


async def test_ids_are_generated_without_caller_input(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.commit()
    assert len(tenant.id) == 26


async def test_tenant_name_is_unique(session: AsyncSession) -> None:
    session.add(Tenant(name="acme"))
    await session.commit()
    session.add(Tenant(name="acme"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_cache_entry_tier_key_is_unique(session: AsyncSession) -> None:
    session.add(CacheEntry(tier=CacheTier.EXACT, key_hash="b" * 64, payload="{}"))
    await session.commit()
    session.add(CacheEntry(tier=CacheTier.EXACT, key_hash="b" * 64, payload="{}"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_same_key_is_allowed_across_tiers(session: AsyncSession) -> None:
    session.add(CacheEntry(tier=CacheTier.EXACT, key_hash="c" * 64, payload="{}"))
    session.add(CacheEntry(tier=CacheTier.PREFIX, key_hash="c" * 64, payload="{}"))
    await session.commit()
    rows = (await session.execute(select(CacheEntry))).scalars().all()
    assert len(rows) == 2


async def test_budget_is_unique_per_tenant_period(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.flush()
    session.add_all(
        [
            Budget(tenant_id=tenant.id, period="monthly", limit_micros=10_000),
            Budget(tenant_id=tenant.id, period="monthly", limit_micros=20_000),
        ]
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_run_cascades_to_nodes_and_events(session: AsyncSession) -> None:
    run = Run(dag_name="nightly", partition="2026-01-14", input_hash="d" * 64)
    session.add(run)
    await session.flush()
    session.add(Node(run_id=run.id, name="extract", status=NodeStatus.SUCCEEDED))
    session.add(RunEvent(run_id=run.id, node_name="extract", event="node_succeeded"))
    await session.commit()

    loaded = (await session.execute(select(Run).options(selectinload(Run.nodes)))).scalar_one()
    assert len(loaded.nodes) == 1

    await session.delete(loaded)
    await session.commit()

    assert (await session.execute(select(Node))).scalars().all() == []
    assert (await session.execute(select(RunEvent))).scalars().all() == []


async def test_nodes_must_be_loaded_explicitly(session: AsyncSession) -> None:
    """Guards the raise_on_sql contract. Silently reverting it to selectin would
    trade a loud error for a MissingGreenlet deep inside unrelated code."""
    run = Run(dag_name="nightly", input_hash="d" * 64)
    session.add(run)
    await session.flush()
    session.add(Node(run_id=run.id, name="extract"))
    await session.commit()

    fetched = (await session.execute(select(Run))).scalar_one()
    with pytest.raises(InvalidRequestError):
        _ = fetched.nodes


async def test_node_name_is_unique_within_a_run(session: AsyncSession) -> None:
    run = Run(dag_name="nightly", input_hash="e" * 64)
    session.add(run)
    await session.flush()
    session.add_all(
        [
            Node(run_id=run.id, name="extract"),
            Node(run_id=run.id, name="extract"),
        ]
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_status_defaults_to_queued(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.flush()
    request = InferenceRequest(
        tenant_id=tenant.id, model="sim-7b", prompt_hash="f" * 64, max_tokens=16
    )
    session.add(request)
    await session.commit()
    await session.refresh(request)
    assert request.status is RequestStatus.QUEUED
    assert request.created_at is not None


async def test_attributes_stay_readable_after_commit(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.commit()
    assert tenant.id and tenant.name == "acme"
