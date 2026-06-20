"""Contradiction detection — emergent from the ledger, not a separate module.

Two active (non-retracted, non-superseded) claims with the same
``(subject, predicate)`` contradict when their valid intervals overlap and either:

* **value_conflict** — different object / object_literal / value qualifier
  (e.g. stake 5.1% vs 9.2% for the *same* period), or
* **polarity_conflict** — one asserted, one negated.

Valid intervals are treated as half-open ``[valid_from, valid_to)`` so that a
claim and its supersessor (which share a boundary date) do not falsely conflict.

:func:`detect_contradictions` is the pure query; :func:`materialize_contradictions`
persists the results into the ``contradiction`` table (idempotent) for the
resolution workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Claim, Contradiction
from clew.ledger.asof import claims_asof

# Qualifiers that carry the claim's "value" (so differing values = a conflict),
# as opposed to descriptive qualifiers (cusip, event_date) that may differ
# without contradicting. Pack-A / ownership-centric.
VALUE_QUALIFIER_KEYS: tuple[str, ...] = ("stake_pct", "shares", "voting_power")


@dataclass(slots=True)
class ContradictionPair:
    claim_a: Claim
    claim_b: Claim
    type: str


def _intervals_overlap(a: Claim, b: Claim) -> bool:
    # Half-open intervals [from, to): None is open-ended (-inf .. +inf).
    # Adjacent intervals that share a boundary (a_to == b_from) do NOT overlap,
    # so a claim and its supersessor are not flagged as contradictions.
    a_from, a_to = a.valid_from, a.valid_to
    b_from, b_to = b.valid_from, b.valid_to
    if a_to is not None and b_from is not None and a_to <= b_from:
        return False
    if b_to is not None and a_from is not None and b_to <= a_from:
        return False
    return True


def detect_contradictions(
    session: Session, *, as_of=None, include_superseded: bool = False
) -> list[ContradictionPair]:
    """Find contradicting claim pairs among active claims (as of ``as_of``).

    Superseded claims (``superseded_by`` set) are historical and excluded by
    default — supersession is precisely how stake-change "conflicts" get resolved.
    """
    claims = claims_asof(session, as_of=as_of)
    if not include_superseded:
        claims = [c for c in claims if c.superseded_by is None]

    # Group by the full relationship (subject, predicate, object). OWNS is
    # non-functional — a filer owns many issuers — so a conflict requires the
    # SAME (subject, predicate, object) asserting incompatible values/polarity
    # over overlapping valid time, not merely a shared subject+predicate.
    def _obj_key(c: Claim):
        lit = None if c.object_literal is None else tuple(sorted(c.object_literal.items()))
        return (c.object_id, lit)

    by_spo: dict[tuple, list[Claim]] = {}
    for c in claims:
        by_spo.setdefault((c.subject_id, c.predicate, _obj_key(c)), []).append(c)

    pairs: list[ContradictionPair] = []
    for group in by_spo.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if not _intervals_overlap(a, b):
                    continue
                if a.polarity != b.polarity:
                    pairs.append(ContradictionPair(a, b, "polarity_conflict"))
                elif _value_qualifiers(a) != _value_qualifiers(b):
                    pairs.append(ContradictionPair(a, b, "value_conflict"))
    return pairs


def _value_qualifiers(c: Claim):
    quals = c.qualifiers or {}
    return tuple((k, quals.get(k)) for k in VALUE_QUALIFIER_KEYS if k in quals)


def materialize_contradictions(session: Session) -> dict:
    """Persist detected contradictions into the ``contradiction`` table (idempotent).

    Existing rows for the same unordered claim pair are not duplicated, so this
    can be re-run after each ingest/reconcile cycle.
    """
    existing: set[frozenset[int]] = {
        frozenset({row.claim_a, row.claim_b})
        for row in session.execute(select(Contradiction)).scalars().all()
    }
    inserted = 0
    for pair in detect_contradictions(session):
        key = frozenset({pair.claim_a.id, pair.claim_b.id})
        if key in existing:
            continue
        existing.add(key)
        session.add(
            Contradiction(
                claim_a=pair.claim_a.id,
                claim_b=pair.claim_b.id,
                type=pair.type,
                status="open",
            )
        )
        inserted += 1
    return {"inserted": inserted, "total_open_pairs": len(existing)}
