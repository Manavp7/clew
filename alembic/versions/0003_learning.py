"""continuous-learning loop: merge suggestions + ER decisions + merge log

Revision ID: 0003_learning
Revises: 0002_entity_seq
Create Date: 2026-06-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_learning"
down_revision: str | None = "0002_entity_seq"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "merge_suggestion",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("entity_a", sa.String(32), nullable=False),
        sa.Column("entity_b", sa.String(32), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("entity_a", "entity_b", name="uq_merge_pair"),
    )
    op.create_index("ix_merge_status", "merge_suggestion", ["status"])

    op.create_table(
        "er_decision",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("key_a", sa.Text, nullable=False),
        sa.Column("key_b", sa.Text, nullable=False),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("key_a", "key_b", name="uq_er_decision_pair"),
    )

    op.create_table(
        "entity_merge_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("kept_id", sa.Text, nullable=False),
        sa.Column("dropped_id", sa.Text, nullable=False),
        sa.Column("dropped_name", sa.Text),
        sa.Column("claims_repointed", sa.Integer, server_default="0"),
        sa.Column("mentions_repointed", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("entity_merge_log")
    op.drop_table("er_decision")
    op.drop_index("ix_merge_status", table_name="merge_suggestion")
    op.drop_table("merge_suggestion")
