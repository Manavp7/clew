"""watchlists + alerts

Revision ID: 0004_alerts
Revises: 0003_learning
Create Date: 2026-06-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_alerts"
down_revision: str | None = "0003_learning"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "watch",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("target", sa.Text),
        sa.Column("threshold", sa.Float),
        sa.Column("label", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "alert",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("watch_id", sa.BigInteger, sa.ForeignKey("watch.id"), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("dedup_key", sa.Text, nullable=False),
        sa.Column("payload", postgresql.JSONB, server_default="{}"),
        sa.Column("seen", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("watch_id", "dedup_key", name="uq_alert_dedup"),
    )


def downgrade() -> None:
    op.drop_table("alert")
    op.drop_table("watch")
