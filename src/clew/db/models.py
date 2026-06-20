"""SQLAlchemy ORM models for the bitemporal, evidence-backed Claim Ledger.

The ledger is the *single source of truth*. It is append-only:

* Claims are never hard-updated. A claim is retracted by setting
  ``retracted_at`` (transaction time) and/or superseded by a newer claim.
* The graph, vector index, and any other store are rebuildable *projections*
  of these tables and never the source of truth.

Two time axes (bitemporal):

* **Valid time** (``valid_from`` / ``valid_to``): when the statement is true in
  the world.
* **Transaction time** (``asserted_at`` / ``retracted_at``): when *we* learned
  or stopped believing it.
"""

from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Embedding dimension. Kept in sync with CLEW_EMBEDDING_DIM via the migration.
EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "source"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    documents: Mapped[list[Document]] = relationship(back_populates="source")


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("source_id", "content_hash", name="uq_document_source_hash"),
        Index("ix_document_doc_type", "doc_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text)  # EDGAR accession no
    doc_type: Mapped[str | None] = mapped_column(Text)  # '13D', '13G', '8-K'
    url: Mapped[str | None] = mapped_column(Text)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    raw_path: Mapped[str | None] = mapped_column(Text)
    # Normalised text: ALL downstream char offsets index into this exact string.
    text: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")

    source: Mapped[Source] = relationship(back_populates="documents")
    mentions: Mapped[list[Mention]] = relationship(back_populates="document")


class Entity(Base):
    """Canonical, resolved entity. ER anchors live in ``external_ids``."""

    __tablename__ = "entity"
    __table_args__ = (
        Index("ix_entity_external_ids", "external_ids", postgresql_using="gin"),
        Index("ix_entity_aliases", "aliases", postgresql_using="gin"),
        Index("ix_entity_type", "type"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # ORG_000123 / PER_000456
    type: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default="{}")
    external_ids: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Mention(Base):
    """Raw, pre-resolution surface mention with exact source offsets."""

    __tablename__ = "mention"
    __table_args__ = (
        Index("ix_mention_resolved_to", "resolved_to"),
        Index("ix_mention_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"), nullable=False)
    surface_text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    ner_type: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    resolved_to: Mapped[str | None] = mapped_column(ForeignKey("entity.id"))
    resolution_confidence: Mapped[float | None] = mapped_column(Float)
    extractor: Mapped[str] = mapped_column(Text, nullable=False)  # 'gliner@<ver>'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="mentions")
    entity: Mapped[Entity | None] = relationship()


class Claim(Base):
    """A statement (not an edge). The heart of the ledger."""

    __tablename__ = "claim"
    __table_args__ = (
        Index("ix_claim_subject_predicate", "subject_id", "predicate"),
        Index("ix_claim_object", "object_id"),
        Index("ix_claim_asserted_at", "asserted_at"),
        Index("ix_claim_spo", "predicate", "subject_id", "object_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("entity.id"), nullable=False)
    predicate: Mapped[str] = mapped_column(Text, nullable=False)
    object_id: Mapped[str | None] = mapped_column(ForeignKey("entity.id"))
    object_literal: Mapped[dict | None] = mapped_column(JSONB)  # {value, unit} for literals
    qualifiers: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")

    # VALID time (true in the world)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)

    # TRANSACTION time (when WE learned / retracted it)
    asserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    retracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    superseded_by: Mapped[int | None] = mapped_column(ForeignKey("claim.id"))

    polarity: Mapped[str] = mapped_column(Text, nullable=False, default="asserted")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    extractor: Mapped[str] = mapped_column(Text, nullable=False)  # model@version
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    evidence: Mapped[list[Evidence]] = relationship(
        back_populates="claim", foreign_keys="Evidence.claim_id"
    )


class Evidence(Base):
    """Exact source span backing a claim (provenance)."""

    __tablename__ = "evidence"
    __table_args__ = (
        Index("ix_evidence_claim", "claim_id"),
        Index("ix_evidence_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claim.id"), nullable=False)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"), nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str | None] = mapped_column(Text)  # 'llm-extract' | 'rule' | 'ner'
    model_version: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    claim: Mapped[Claim] = relationship(back_populates="evidence", foreign_keys=[claim_id])


class Contradiction(Base):
    """Emergent conflict between two claims (materialised by a detector job)."""

    __tablename__ = "contradiction"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_a: Mapped[int] = mapped_column(ForeignKey("claim.id"), nullable=False)
    claim_b: Mapped[int] = mapped_column(ForeignKey("claim.id"), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    resolution: Mapped[str | None] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MergeSuggestion(Base):
    """A candidate entity merge awaiting human review (continuous-learning loop)."""

    __tablename__ = "merge_suggestion"
    __table_args__ = (
        UniqueConstraint("entity_a", "entity_b", name="uq_merge_pair"),
        Index("ix_merge_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Plain string refs (not FKs): an accepted merge deletes one entity, but we
    # keep the suggestion row for audit — like entity_merge_log.
    entity_a: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_b: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    # status: open | accepted | rejected
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ERDecision(Base):
    """Persisted human ER feedback: must-link / must-not-link keyed by core name.

    Consulted by the resolver so confirmed merges become deterministic and
    rejected merges are permanently blocked — the loop that lets human
    corrections compound over time.
    """

    __tablename__ = "er_decision"
    __table_args__ = (UniqueConstraint("key_a", "key_b", name="uq_er_decision_pair"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key_a: Mapped[str] = mapped_column(Text, nullable=False)  # normalized core name
    key_b: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)  # must_link | must_not_link
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EntityMergeLog(Base):
    """Audit trail of executed entity merges (kept_id absorbed dropped_id)."""

    __tablename__ = "entity_merge_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kept_id: Mapped[str] = mapped_column(Text, nullable=False)
    dropped_id: Mapped[str] = mapped_column(Text, nullable=False)
    dropped_name: Mapped[str | None] = mapped_column(Text)
    claims_repointed: Mapped[int] = mapped_column(Integer, default=0)
    mentions_repointed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Watch(Base):
    """A saved watch that generates alerts when the ledger changes."""

    __tablename__ = "watch"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # kind: stake_threshold | new_claim | contradiction
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    target: Mapped[str | None] = mapped_column(Text)  # entity id (or None for global)
    threshold: Mapped[float | None] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Alert(Base):
    """A fired alert (deduped per watch by ``dedup_key``)."""

    __tablename__ = "alert"
    __table_args__ = (UniqueConstraint("watch_id", "dedup_key", name="uq_alert_dedup"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    watch_id: Mapped[int] = mapped_column(ForeignKey("watch.id"), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    dedup_key: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    seen: Mapped[bool] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvalRun(Base):
    """Versioned evaluation run — metrics over time vs pipeline/model version."""

    __tablename__ = "eval_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stage: Mapped[str] = mapped_column(Text, nullable=False)  # parsing|extraction|er|reasoning
    dataset: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_git_sha: Mapped[str | None] = mapped_column(Text)
    model_versions: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
