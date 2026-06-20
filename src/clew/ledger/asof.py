"""Bitemporal query helpers — the "for free" payoff of the ledger model.

* **As-of (transaction time)**: what did we believe at time ``T``?
* **Valid-as-of (world time)**: which claims were true in the world on date ``D``?
* **Knowledge diff**: what changed between two transaction-time instants?
* **Timeline**: ordered history of a (subject, predicate[, object]) — drives the
  stake-change timeline for the 13D/13G wedge.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from clew.db.models import Claim


def _asof_predicate(as_of: datetime | None):
    """Claims believed as of transaction time ``as_of`` (None = now / latest)."""
    if as_of is None:
        return Claim.retracted_at.is_(None)
    return and_(
        Claim.asserted_at <= as_of,
        or_(Claim.retracted_at.is_(None), Claim.retracted_at > as_of),
    )


def claims_asof(
    session: Session,
    *,
    as_of: datetime | None = None,
    valid_on: date | None = None,
    subject_id: str | None = None,
    predicate: str | None = None,
    object_id: str | None = None,
) -> list[Claim]:
    """Return claims believed as of ``as_of`` and (optionally) valid on ``valid_on``."""
    stmt = select(Claim).where(_asof_predicate(as_of))
    if valid_on is not None:
        stmt = stmt.where(
            or_(Claim.valid_from.is_(None), Claim.valid_from <= valid_on),
            or_(Claim.valid_to.is_(None), Claim.valid_to >= valid_on),
        )
    if subject_id is not None:
        stmt = stmt.where(Claim.subject_id == subject_id)
    if predicate is not None:
        stmt = stmt.where(Claim.predicate == predicate)
    if object_id is not None:
        stmt = stmt.where(Claim.object_id == object_id)
    return list(session.execute(stmt).scalars().all())


def knowledge_diff(session: Session, since: datetime, until: datetime | None = None):
    """What did we learn between ``since`` and ``until``?

    Returns ``(asserted, retracted)`` lists of claims whose transaction-time
    events fall in the window — answers "what changed this week?".
    """
    until = until or datetime.max
    asserted = list(
        session.execute(
            select(Claim).where(and_(Claim.asserted_at >= since, Claim.asserted_at <= until))
        )
        .scalars()
        .all()
    )
    retracted = list(
        session.execute(
            select(Claim).where(
                and_(Claim.retracted_at.is_not(None), Claim.retracted_at >= since,
                     Claim.retracted_at <= until)
            )
        )
        .scalars()
        .all()
    )
    return asserted, retracted


def timeline(
    session: Session,
    *,
    subject_id: str,
    predicate: str,
    object_id: str | None = None,
    as_of: datetime | None = None,
) -> list[Claim]:
    """Chronological history of a relationship, ordered by valid time then assertion.

    For ``predicate=OWNS`` this is the stake-change timeline: each row carries
    ``qualifiers.stake_pct`` and a valid interval, so consumers can render how an
    ownership stake evolved over time.
    """
    stmt = (
        select(Claim)
        .where(_asof_predicate(as_of))
        .where(Claim.subject_id == subject_id, Claim.predicate == predicate)
    )
    if object_id is not None:
        stmt = stmt.where(Claim.object_id == object_id)
    stmt = stmt.order_by(Claim.valid_from.nulls_last(), Claim.asserted_at)
    return list(session.execute(stmt).scalars().all())
