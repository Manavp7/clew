"""Contradiction detection — emergent from the ledger, not a separate module.

Two non-retracted claims with the same ``(subject, predicate)`` contradict when
their valid intervals overlap and either:

* **value_conflict** — different object / object_literal (e.g. stake 5.1% vs 9.2%
  for the same period), or
* **polarity_conflict** — one asserted, one negated.

This is a pure query over the ledger; Phase 2 materialises results into the
``contradiction`` table and adds resolution workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from clew.db.models import Claim
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
    # Treat None as open-ended (-inf .. +inf).
    a_from, a_to = a.valid_from, a.valid_to
    b_from, b_to = b.valid_from, b.valid_to
    if a_to is not None and b_from is not None and a_to < b_from:
        return False
    if b_to is not None and a_from is not None and b_to < a_from:
        return False
    return True


def _value_key(c: Claim):
    """The comparable 'value' of a claim: object + value-bearing qualifiers."""
    literal = None if c.object_literal is None else tuple(sorted(c.object_literal.items()))
    quals = c.qualifiers or {}
    value_quals = tuple((k, quals.get(k)) for k in VALUE_QUALIFIER_KEYS if k in quals)
    return (c.object_id, literal, value_quals)


def detect_contradictions(session: Session, *, as_of=None) -> list[ContradictionPair]:
    """Find contradicting claim pairs currently believed (as of ``as_of``)."""
    claims = claims_asof(session, as_of=as_of)
    by_sp: dict[tuple[str, str], list[Claim]] = {}
    for c in claims:
        by_sp.setdefault((c.subject_id, c.predicate), []).append(c)

    pairs: list[ContradictionPair] = []
    for group in by_sp.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if not _intervals_overlap(a, b):
                    continue
                if a.polarity != b.polarity:
                    pairs.append(ContradictionPair(a, b, "polarity_conflict"))
                elif _value_key(a) != _value_key(b):
                    pairs.append(ContradictionPair(a, b, "value_conflict"))
    return pairs
