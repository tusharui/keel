from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from keel.config import get_settings


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """SQLite parses and ignores ``ondelete`` unless the pragma is on.

    Left off, referential integrity is unenforced in development and enforced in
    production, which means every cascading delete bug reproduces only after
    deploy. Turning it on costs one statement per connection.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def build_engine(url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    resolved = url or get_settings().database_url

    if resolved.startswith("sqlite"):
        # An in-memory database lives inside a single connection, so pooling it
        # hands out empty databases. StaticPool pins every session to the one
        # connection that owns the data.
        engine = create_async_engine(
            resolved,
            echo=echo,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        _enable_sqlite_foreign_keys(engine)
        return engine

    return create_async_engine(
        resolved,
        echo=echo,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False keeps attributes readable after commit, so a
    # response can be serialised from an ORM object without re-querying it.
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
