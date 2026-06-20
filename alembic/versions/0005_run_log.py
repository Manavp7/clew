"""monitoring: run_log

Revision ID: 0005_run_log
Revises: 0004_alerts
Create Date: 2026-06-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_run_log"
down_revision: str | None = "0004_alerts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="ok"),
        sa.Column("counts", postgresql.JSONB, server_default="{}"),
        sa.Column("duration_ms", sa.Integer, server_default="0"),
        sa.Column("pipeline_git_sha", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_run_log_stage", "run_log", ["stage"])


def downgrade() -> None:
    op.drop_index("ix_run_log_stage", table_name="run_log")
    op.drop_table("run_log")
