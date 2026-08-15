"""ER continuous-learning loop: suggest merges, record human decisions.

* :func:`generate_suggestions` — surface near-threshold same-type entity pairs
  (Jaro-Winkler in the "grey zone" just below the auto-merge cutoff) that a human
  should adjudicate. CIK-anchored entities with *different* CIKs are never
  suggested (authoritative split).
* :func:`accept_suggestion` — execute the merge and record a ``must_link`` decision.
* :func:`reject_suggestion` — record a ``must_not_link`` decision so the pair is
  never re-suggested or auto-merged.

Decisions are keyed by normalized core name and consulted by the resolver, so
human corrections compound across future runs.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rapidfuzz import distance
from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Entity, ERDecision, MergeSuggestion
from clew.resolve.merge import _entity_claim_count, merge_entities
from clew.resolve.normalize import block_key, core, normalized

GREY_LOW = 0.85
GREY_HIGH = 0.92  # resolver auto-merges at/above this; below GREY_LOW we ignore
HIGH_IMPACT_CLAIMS = 5  # entities with >= this many claims require explicit human accept


def _jw(a: str, b: str) -> float:
    return 1.0 - distance.JaroWinkler.normalized_distance(a, b)


def _decision_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((core(a), core(b))))  # type: ignore[return-value]


def generate_suggestions(session: Session, *, limit: int = 200) -> dict:
    entities = session.execute(select(Entity)).scalars().all()
    norms = {e.id: normalized(e.canonical_name) for e in entities}

    decided: set[tuple[str, str]] = {
        (d.key_a, d.key_b) for d in session.execute(select(ERDecision)).scalars().all()
    }
    existing: set[tuple[str, str]] = {
        tuple(sorted((s.entity_a, s.entity_b)))  # type: ignore[misc]
        for s in session.execute(select(MergeSuggestion)).scalars().all()
    }

    # Block by (type, blocking-key) to keep comparisons cheap.
    blocks: dict[tuple[str, str], list[Entity]] = {}
    for e in entities:
        blocks.setdefault((e.type, block_key(e.canonical_name)), []).append(e)

    created = 0
    for group in blocks.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ca, cb = (a.external_ids or {}).get("cik"), (b.external_ids or {}).get("cik")
                if ca and cb and ca != cb:
                    continue  # authoritative CIK split
                score = _jw(norms[a.id], norms[b.id])
                if not (GREY_LOW <= score < GREY_HIGH):
                    continue
                if _decision_key(a.canonical_name, b.canonical_name) in decided:
                    continue
                pair = tuple(sorted((a.id, b.id)))
                if pair in existing:
                    continue
                existing.add(pair)
                session.add(
                    MergeSuggestion(
                        entity_a=pair[0],
                        entity_b=pair[1],
                        score=round(score, 4),
                        reason=f"name similarity {score:.3f} in grey zone",
                    )
                )
                created += 1
                if created >= limit:
                    return {"suggested": created}
    return {"suggested": created}


def accept_suggestion(session: Session, suggestion_id: int) -> dict:
    s = session.get(MergeSuggestion, suggestion_id)
    if s is None:
        raise ValueError(f"suggestion {suggestion_id} not found")
    keep, drop = _choose_keep_drop(session, s.entity_a, s.entity_b)
    keep_name = session.get(Entity, keep).canonical_name
    drop_name = session.get(Entity, drop).canonical_name
    result = merge_entities(session, keep, drop)
    _record_decision(session, keep_name, drop_name, "must_link")
    s.status = "accepted"
    s.decided_at = datetime.now(UTC)
    session.flush()
    return result


def reject_suggestion(session: Session, suggestion_id: int) -> dict:
    s = session.get(MergeSuggestion, suggestion_id)
    if s is None:
        raise ValueError(f"suggestion {suggestion_id} not found")
    a, b = session.get(Entity, s.entity_a), session.get(Entity, s.entity_b)
    _record_decision(session, a.canonical_name, b.canonical_name, "must_not_link")
    s.status = "rejected"
    s.decided_at = datetime.now(UTC)
    session.flush()
    return {"rejected": suggestion_id}


def load_decisions(session: Session) -> tuple[set, set]:
    """Return (must_link, must_not_link) sets of frozenset{core_a, core_b}."""
    must_link, must_not = set(), set()
    for d in session.execute(select(ERDecision)).scalars().all():
        key = frozenset({d.key_a, d.key_b})
        (must_link if d.decision == "must_link" else must_not).add(key)
    return must_link, must_not


def _choose_keep_drop(session: Session, a: str, b: str) -> tuple[str, str]:
    # Keep the entity with more claims (more established); tie-break by id.
    na, nb = _entity_claim_count(session, a), _entity_claim_count(session, b)
    return (a, b) if (na, a) >= (nb, b) else (b, a)


def _record_decision(session: Session, name_a: str, name_b: str, decision: str) -> None:
    ka, kb = _decision_key(name_a, name_b)
    existing = session.execute(
        select(ERDecision).where(ERDecision.key_a == ka, ERDecision.key_b == kb)
    ).scalars().first()
    if existing is None:
        session.add(ERDecision(key_a=ka, key_b=kb, decision=decision))
    else:
        existing.decision = decision
