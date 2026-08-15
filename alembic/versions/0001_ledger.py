"""initial bitemporal claim ledger

Revision ID: 0001_ledger
Revises:
Create Date: 2026-06-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from clew.config import get_settings

revision: str = "0001_ledger"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = get_settings().embedding_dim


def upgrade() -> None:
    # --- extensions ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # --- source ---
    op.create_table(
        "source",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("url", sa.Text),
        sa.Column("publisher", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- document ---
    op.create_table(
        "document",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.BigInteger, sa.ForeignKey("source.id"), nullable=False),
        sa.Column("external_id", sa.Text),
        sa.Column("doc_type", sa.Text),
        sa.Column("url", sa.Text),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column("raw_path", sa.Text),
        sa.Column("text", sa.Text),
        sa.Column("meta", postgresql.JSONB, server_default="{}"),
        sa.UniqueConstraint("source_id", "content_hash", name="uq_document_source_hash"),
    )
    op.create_index("ix_document_doc_type", "document", ["doc_type"])

    # --- entity ---
    op.create_table(
        "entity",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("canonical_name", sa.Text, nullable=False),
        sa.Column("aliases", postgresql.ARRAY(sa.Text), server_default="{}"),
        sa.Column("external_ids", postgresql.JSONB, server_default="{}"),
        sa.Column("embedding", Vector(DIM)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_entity_external_ids", "entity", ["external_ids"], postgresql_using="gin")
    op.create_index("ix_entity_aliases", "entity", ["aliases"], postgresql_using="gin")
    op.create_index("ix_entity_type", "entity", ["type"])

    # --- mention ---
    op.create_table(
        "mention",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.BigInteger, sa.ForeignKey("document.id"), nullable=False),
        sa.Column("surface_text", sa.Text, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("ner_type", sa.Text),
        sa.Column("embedding", Vector(DIM)),
        sa.Column("resolved_to", sa.String(32), sa.ForeignKey("entity.id")),
        sa.Column("resolution_confidence", sa.Float),
        sa.Column("extractor", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_mention_resolved_to", "mention", ["resolved_to"])
    op.create_index("ix_mention_document", "mention", ["document_id"])

    # --- claim ---
    op.create_table(
        "claim",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("subject_id", sa.String(32), sa.ForeignKey("entity.id"), nullable=False),
        sa.Column("predicate", sa.Text, nullable=False),
        sa.Column("object_id", sa.String(32), sa.ForeignKey("entity.id")),
        sa.Column("object_literal", postgresql.JSONB),
        sa.Column("qualifiers", postgresql.JSONB, server_default="{}"),
        sa.Column("valid_from", sa.Date),
        sa.Column("valid_to", sa.Date),
        sa.Column("asserted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("retracted_at", sa.DateTime(timezone=True)),
        sa.Column("superseded_by", sa.BigInteger, sa.ForeignKey("claim.id")),
        sa.Column("polarity", sa.Text, nullable=False, server_default="asserted"),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("extractor", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_claim_subject_predicate", "claim", ["subject_id", "predicate"])
    op.create_index("ix_claim_object", "claim", ["object_id"])
    op.create_index("ix_claim_asserted_at", "claim", ["asserted_at"])
    op.create_index("ix_claim_spo", "claim", ["predicate", "subject_id", "object_id"])

    # --- evidence ---
    op.create_table(
        "evidence",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("claim_id", sa.BigInteger, sa.ForeignKey("claim.id"), nullable=False),
        sa.Column("document_id", sa.BigInteger, sa.ForeignKey("document.id"), nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("snippet", sa.Text, nullable=False),
        sa.Column("method", sa.Text),
        sa.Column("model_version", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_evidence_claim", "evidence", ["claim_id"])
    op.create_index("ix_evidence_document", "evidence", ["document_id"])

    # --- contradiction ---
    op.create_table(
        "contradiction",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("claim_a", sa.BigInteger, sa.ForeignKey("claim.id"), nullable=False),
        sa.Column("claim_b", sa.BigInteger, sa.ForeignKey("claim.id"), nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
        sa.Column("resolution", sa.Text),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- eval_run ---
    op.create_table(
        "eval_run",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("dataset", sa.Text, nullable=False),
        sa.Column("pipeline_git_sha", sa.Text),
        sa.Column("model_versions", postgresql.JSONB),
        sa.Column("metrics", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("eval_run")
    op.drop_table("contradiction")
    op.drop_table("evidence")
    op.drop_table("claim")
    op.drop_table("mention")
    op.drop_table("entity")
    op.drop_index("ix_document_doc_type", table_name="document")
    op.drop_table("document")
    op.drop_table("source")
