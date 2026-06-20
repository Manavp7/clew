"""Pydantic schemas for structured claim extraction.

``ExtractedClaim`` is the provider-agnostic intermediate representation produced
by *any* extractor (rule-based or LLM). The pipeline then:

1. grounds ``evidence_quote`` to exact char offsets in ``document.text``;
2. resolves ``subject_surface`` / ``object_surface`` to canonical entity ids;
3. writes a ledger :class:`~clew.ledger.writer.ClaimInput`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from clew.packs.pack_a_financial import Predicate


class ExtractedClaim(BaseModel):
    subject_surface: str = Field(description="Verbatim name of the subject entity")
    predicate: str = Field(description=f"One of: {[p.value for p in Predicate]}")
    object_surface: str | None = Field(
        default=None, description="Verbatim name of the object entity, if any"
    )
    object_literal: dict | None = Field(
        default=None, description="Literal object {value, unit} when not an entity"
    )
    qualifiers: dict = Field(
        default_factory=dict,
        description="e.g. {stake_pct, shares, security_class, event_date}",
    )
    valid_from: str | None = Field(default=None, description="ISO date the statement becomes true")
    polarity: str = Field(default="asserted", description="'asserted' or 'negated'")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence_quote: str = Field(
        description="Verbatim span from the document that supports this claim"
    )


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)
