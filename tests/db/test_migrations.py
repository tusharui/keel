from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from keel.db.base import Base

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _migrate(url: str) -> None:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")


def test_migrations_build_the_current_schema(tmp_path: Path) -> None:
    """Fails the moment a model changes without a matching migration.

    Autogenerate silently produces no diff when it cannot represent a change,
    so this is the only check that catches model and migration drifting apart.
    """
    db_path = tmp_path / "migrated.db"
    _migrate(f"sqlite+aiosqlite:///{db_path}")

    sync_engine = create_engine(f"sqlite:///{db_path}")
    with sync_engine.connect() as connection:
        context = MigrationContext.configure(connection)
        diff = compare_metadata(context, Base.metadata)

    assert diff == [], f"models and migrations diverged: {diff}"


def test_schema_downgrades_to_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "down.db"
    _migrate(f"sqlite+aiosqlite:///{db_path}")

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")
    command.downgrade(config, "base")


@pytest.mark.parametrize("revision", ["a1b2c3d4e5f6"])
def test_each_revision_is_reachable_from_scratch(tmp_path: Path, revision: str) -> None:
    db_path = tmp_path / f"{revision}.db"
    _migrate(f"sqlite+aiosqlite:///{db_path}")

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")
    command.downgrade(config, "-1")
    command.upgrade(config, revision)
