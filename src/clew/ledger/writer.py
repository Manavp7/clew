"""Append-only writers for the Claim Ledger.

Invariants enforced here (the ledger's guarantees):

* **Every claim has >= 1 evidence span.** A claim with no evidence is rejected.
* **Every claim records its extractor** (``model@version``) for reproducibility.
* **Evidence offsets round-trip**: ``document.text[char_start:char_end]`` must
  equal the stored snippet. This keeps provenance exact and rebuildable.
* **Append-only**: claims are never hard-deleted. Use :func:`retract_claim`
  (transaction-time retraction) or :func:`supersede_claim` (new claim replaces
  an old one) instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Claim, Document, Evidence, Mention


class LedgerError(ValueError):
    """Raised when an append-only / provenance invariant is violated."""


@dataclass(slots=True)
class EvidenceInput:
    document_id: int
    char_start: int
    char_end: int
    snippet: str
    method: str | None = None
    model_version: str | None = None


@dataclass(slots=True)
class ClaimInput:
    subject_id: str
    predicate: str
    extractor: str
    object_id: str | None = None
    object_literal: dict | None = None
    qualifiers: dict = field(default_factory=dict)
    valid_from: date | None = None
    valid_to: date | None = None
    polarity: str = "asserted"
    confidence: float = 1.0
    evidence: list[EvidenceInput] = field(default_factory=list)


def write_mention(
    session: Session,
    *,
    document_id: int,
    surface_text: str,
    char_start: int,
    char_end: int,
    extractor: str,
    ner_type: str | None = None,
    resolved_to: str | None = None,
    resolution_confidence: float | None = None,
    verify_offsets: bool = True,
) -> Mention:
    if verify_offsets:
        _verify_offsets(session, document_id, char_start, char_end, surface_text)
    mention = Mention(
        document_id=document_id,
        surface_text=surface_text,
        char_start=char_start,
        char_end=char_end,
        ner_type=ner_type,
        resolved_to=resolved_to,
        resolution_confidence=resolution_confidence,
        extractor=extractor,
    )
    session.add(mention)
    session.flush()
    return mention


def write_claim(session: Session, claim_in: ClaimInput, *, verify_offsets: bool = True) -> Claim:
    """Append a claim with its evidence. Rejects claims lacking evidence."""
    if not claim_in.evidence:
        raise LedgerError(
            f"claim {claim_in.subject_id} {claim_in.predicate} has no evidence; "
            "every claim must cite >= 1 source span"
        )
    if not claim_in.extractor:
        raise LedgerError("claim is missing extractor (model@version) metadata")

    for ev in claim_in.evidence:
        if verify_offsets:
            _verify_offsets(session, ev.document_id, ev.char_start, ev.char_end, ev.snippet)

    claim = Claim(
        subject_id=claim_in.subject_id,
        predicate=claim_in.predicate,
        object_id=claim_in.object_id,
        object_literal=claim_in.object_literal,
        qualifiers=claim_in.qualifiers or {},
        valid_from=claim_in.valid_from,
        valid_to=claim_in.valid_to,
        polarity=claim_in.polarity,
        confidence=claim_in.confidence,
        extractor=claim_in.extractor,
    )
    session.add(claim)
    session.flush()  # assigns claim.id

    for ev in claim_in.evidence:
        session.add(
            Evidence(
                claim_id=claim.id,
                document_id=ev.document_id,
                char_start=ev.char_start,
                char_end=ev.char_end,
                snippet=ev.snippet,
                method=ev.method,
                model_version=ev.model_version,
            )
        )
    session.flush()
    return claim


def retract_claim(session: Session, claim_id: int, *, when: datetime | None = None) -> Claim:
    """Transaction-time retraction: mark a claim no longer believed as of ``when``."""
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise LedgerError(f"claim {claim_id} not found")
    if claim.retracted_at is None:
        claim.retracted_at = when or datetime.now(UTC)
    return claim


def supersede_claim(
    session: Session, old_claim_id: int, new_claim: ClaimInput, *, verify_offsets: bool = True
) -> Claim:
    """Append ``new_claim`` and mark ``old_claim_id`` superseded + retracted."""
    created = write_claim(session, new_claim, verify_offsets=verify_offsets)
    old = session.get(Claim, old_claim_id)
    if old is None:
        raise LedgerError(f"claim {old_claim_id} not found")
    old.superseded_by = created.id
    if old.retracted_at is None:
        old.retracted_at = datetime.now(UTC)
    session.flush()
    return created


def _verify_offsets(
    session: Session, document_id: int, char_start: int, char_end: int, snippet: str
) -> None:
    if char_start < 0 or char_end < char_start:
        raise LedgerError(f"invalid offsets [{char_start}:{char_end}]")
    doc_text = session.execute(
        select(Document.text).where(Document.id == document_id)
    ).scalar_one_or_none()
    if doc_text is None:
        raise LedgerError(f"document {document_id} has no normalized text to anchor offsets")
    actual = doc_text[char_start:char_end]
    if actual != snippet:
        raise LedgerError(
            "evidence offset mismatch: "
            f"text[{char_start}:{char_end}]={actual!r} != snippet={snippet!r}"
        )
