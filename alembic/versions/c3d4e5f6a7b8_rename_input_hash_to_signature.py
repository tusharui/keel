"""rename nodes.input_hash to nodes.signature

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-01-21

The original column name described a hash without saying what it hashed. It now
carries the node's content signature, and the new index supports the
skip-unchanged lookup that finds a matching success in an earlier run.

Added as a new revision rather than by editing the applied one: the baseline
migration has already run against real databases.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Two passes rather than one. Batch mode gathers indexes from both the
    # existing and the target table when it rebuilds, so creating an index for a
    # column being renamed in the same batch looks the name up in a table that
    # does not have it yet.
    with op.batch_alter_table("nodes") as batch:
        batch.alter_column(
            "input_hash", new_column_name="signature", existing_type=sa.String(length=64)
        )
    with op.batch_alter_table("nodes") as batch:
        batch.create_index("ix_nodes_signature", ["signature"])


def downgrade() -> None:
    with op.batch_alter_table("nodes") as batch:
        batch.drop_index("ix_nodes_signature")
    with op.batch_alter_table("nodes") as batch:
        batch.alter_column(
            "signature", new_column_name="input_hash", existing_type=sa.String(length=64)
        )
