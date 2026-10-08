from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


def naming_convention() -> dict[str, str]:
    """Deterministic constraint names.

    Alembic autogenerate produces anonymous constraints without a convention,
    which makes it impossible to tell in a diff whether a constraint was added
    or dropped.
    """
    return {
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=naming_convention())
