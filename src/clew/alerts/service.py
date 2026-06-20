"""Alert generation over the ledger (idempotent, deduped).

Watches express interest in changes; running the generator scans the current
ledger and fires alerts, deduped per watch by a content key so re-runs don't
duplicate. Supported kinds:

* ``stake_threshold`` — fire when a current OWNS claim on the target issuer has
  ``stake_pct >= threshold`` (e.g. "alert when anyone crosses 10% of OceanPal").
* ``new_claim`` — fire for each current claim involving the target entity
  (subject or object).
* ``contradiction`` — fire for each open contradiction (optionally involving the
  target entity).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Alert, Claim, Contradiction, Entity, Watch
from clew.db.session import write_session
from clew.ledger.asof import claims_asof


def _emit(session: Session, watch: Watch, dedup_key: str, message: str, payload: dict) -> bool:
    exists = session.execute(
        select(Alert).where(Alert.watch_id == watch.id, Alert.dedup_key == dedup_key)
    ).scalars().first()
    if exists is not None:
        return False
    session.add(
        Alert(watch_id=watch.id, dedup_key=dedup_key, message=message, payload=payload)
    )
    return True


def _name(session: Session, entity_id: str | None) -> str:
    if not entity_id:
        return "?"
    e = session.get(Entity, entity_id)
    return e.canonical_name if e else entity_id


def run_alerts(session: Session | None = None) -> dict:
    """Scan the ledger and fire alerts. If ``session`` is given, use it (caller
    controls the transaction — handy for tests); otherwise open a write session."""
    if session is not None:
        return {"alerts_fired": _run_all(session)}
    with write_session() as own:
        return {"alerts_fired": _run_all(own)}


def _run_all(session: Session) -> int:
    fired = 0
    for w in session.execute(select(Watch)).scalars().all():
        if w.kind == "stake_threshold":
            fired += _run_stake_threshold(session, w)
        elif w.kind == "new_claim":
            fired += _run_new_claim(session, w)
        elif w.kind == "contradiction":
            fired += _run_contradiction(session, w)
    return fired


def _run_stake_threshold(session: Session, w: Watch) -> int:
    thr = w.threshold or 0.0
    # Current OWNS claims on the target issuer (object).
    claims = claims_asof(session, predicate="OWNS", object_id=w.target)
    n = 0
    for c in claims:
        stake = (c.qualifiers or {}).get("stake_pct")
        if stake is not None and float(stake) >= thr:
            msg = (
                f"{_name(session, c.subject_id)} holds {stake}% of "
                f"{_name(session, c.object_id)} (>= {thr}%)"
            )
            if _emit(session, w, f"claim:{c.id}", msg, {"claim_id": c.id, "stake_pct": stake}):
                n += 1
    return n


def _run_new_claim(session: Session, w: Watch) -> int:
    stmt = select(Claim).where(Claim.retracted_at.is_(None))
    if w.target:
        stmt = stmt.where((Claim.subject_id == w.target) | (Claim.object_id == w.target))
    n = 0
    for c in session.execute(stmt).scalars().all():
        msg = (
            f"{_name(session, c.subject_id)} {c.predicate} "
            f"{_name(session, c.object_id)}"
        )
        if _emit(session, w, f"claim:{c.id}", msg, {"claim_id": c.id}):
            n += 1
    return n


def _run_contradiction(session: Session, w: Watch) -> int:
    rows = session.execute(
        select(Contradiction).where(Contradiction.status == "open")
    ).scalars().all()
    n = 0
    for r in rows:
        if w.target:
            ca, cb = session.get(Claim, r.claim_a), session.get(Claim, r.claim_b)
            involved = w.target in {
                ca.subject_id, ca.object_id, cb.subject_id, cb.object_id
            }
            if not involved:
                continue
        msg = f"{r.type} between claims {r.claim_a} and {r.claim_b}"
        if _emit(session, w, f"contradiction:{r.id}", msg, {"contradiction_id": r.id}):
            n += 1
    return n
