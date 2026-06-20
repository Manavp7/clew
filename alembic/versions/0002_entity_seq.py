"""entity id sequence

Revision ID: 0002_entity_seq
Revises: 0001_ledger
Create Date: 2026-06-20
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_entity_seq"
down_revision: str | None = "0001_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Single global counter; the type prefix (ORG_/PER_/SEC_) encodes the type.
    op.execute("CREATE SEQUENCE IF NOT EXISTS entity_id_seq START 1")


def downgrade() -> None:
    op.execute("DROP SEQUENCE IF EXISTS entity_id_seq")
