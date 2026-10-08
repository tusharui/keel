from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from keel.db.base import Base
from keel.enums import CacheTier, NodeStatus, RequestStatus, RunStatus
from keel.ids import new_id

ULID = String(26)
HASH = String(64)


def _enum(python_enum: type[StrEnum]) -> SAEnum:
    """VARCHAR plus CHECK rather than a native Postgres enum type.

    Native enums have to be altered type-by-type to add a value, which SQLite
    cannot do at all. Keeping both drivers on a plain VARCHAR means one schema
    definition and one migration path.
    """
    return SAEnum(
        python_enum,
        values_callable=lambda members: [m.value for m in members],
        native_enum=False,
        validate_strings=True,
        length=32,
    )


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Budget(Base):
    """Spend cap held in integer microdollars.

    Money never goes through a float. Summing fractional dollars across a busy
    month drifts, and the drift always shows up as a customer overage.
    """

    __tablename__ = "budgets"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ULID, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    period: Mapped[str] = mapped_column(String(16), nullable=False, default="monthly")
    limit_micros: Mapped[int] = mapped_column(BigInteger, nullable=False)
    spent_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    __table_args__ = (Index("uq_budgets_tenant_period", "tenant_id", "period", unique=True),)


class InferenceRequest(Base):
    """One row per admitted request, written once the scheduler finishes it.

    Prompt and completion text are deliberately absent. Keeping the payload out
    of this table means the audit trail can be retained long after the content
    itself should have aged out, and ``prompt_hash`` is enough to correlate a
    request with a cache tier or an eval case.
    """

    __tablename__ = "inference_requests"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ULID, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(HASH, nullable=False)
    status: Mapped[RequestStatus] = mapped_column(
        _enum(RequestStatus), nullable=False, default=RequestStatus.QUEUED, index=True
    )
    cache_tier: Mapped[CacheTier] = mapped_column(
        _enum(CacheTier), nullable=False, default=CacheTier.MISS
    )

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_micros: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tpot_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    queue_wait_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_inference_requests_tenant_created", "tenant_id", "created_at"),)


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    tier: Mapped[CacheTier] = mapped_column(_enum(CacheTier), nullable=False)
    key_hash: Mapped[str] = mapped_column(HASH, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_access_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("uq_cache_entries_tier_key", "tier", "key_hash", unique=True),
        Index("ix_cache_entries_lru", "tier", "last_access_at"),
        Index("ix_cache_entries_expiry", "expires_at"),
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    dag_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    partition: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_hash: Mapped[str] = mapped_column(HASH, nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus), nullable=False, default=RunStatus.PENDING, index=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    nodes: Mapped[list[Node]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        # An AsyncSession cannot run an implicit load from a bare attribute
        # access; it raises MissingGreenlet at runtime, far from the query that
        # should have declared the join. Failing loudly here instead keeps the IO
        # inside an awaited query and makes N+1 visible where it is introduced.
        lazy="raise_on_sql",
    )

    __table_args__ = (
        Index("uq_runs_dag_partition", "dag_name", "partition", "input_hash", unique=True),
    )


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ULID, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[NodeStatus] = mapped_column(
        _enum(NodeStatus), nullable=False, default=NodeStatus.PENDING
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_hash: Mapped[str] = mapped_column(HASH, nullable=False, default="")
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[Run] = relationship(back_populates="nodes")

    __table_args__ = (Index("uq_nodes_run_name", "run_id", "name", unique=True),)


class RunEvent(Base):
    """Append-only. Explains why a node ran, which is the question you ask
    during an incident and the one a mutable status column cannot answer."""

    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ULID, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    node_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    suite: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Span(Base):
    __tablename__ = "spans"

    id: Mapped[str] = mapped_column(ULID, primary_key=True, default=new_id)
    trace_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    parent_id: Mapped[str | None] = mapped_column(ULID, nullable=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_spans_trace_created", "trace_id", "created_at"),)


ALL_MODELS = (
    Tenant,
    Budget,
    InferenceRequest,
    CacheEntry,
    Run,
    Node,
    RunEvent,
    EvalResult,
    Span,
)
