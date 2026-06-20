"""Integration tests for supersession reconciliation + contradiction materialization."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select

from clew.db.models import Contradiction, Document, Source
from clew.db.repo import upsert_entity_by_external_id
from clew.ledger.asof import claims_asof, timeline
from clew.ledger.contradiction import detect_contradictions, materialize_contradictions
from clew.ledger.supersession import reconcile_supersessions
from clew.ledger.writer import ClaimInput, EvidenceInput, write_claim
from clew.packs.pack_a_financial import EntityType, Predicate
from tests.conftest import requires_db

pytestmark = requires_db


def _doc(session, body: str) -> Document:
    src = Source(name="TEST")
    session.add(src)
    session.flush()
    doc = Document(
        source_id=src.id,
        doc_type="13D",
        retrieved_at=datetime.now(UTC),
        content_hash=f"hash-{datetime.now(UTC).timestamp()}",
        text=body,
    )
    session.add(doc)
    session.flush()
    return doc


def _ent(session, name, cik):
    return upsert_entity_by_external_id(
        session,
        entity_type=EntityType.ORGANIZATION,
        canonical_name=name,
        namespace="cik",
        value=cik,
    )


def _owns(session, doc, subj, obj, pct, vfrom):
    body = doc.text
    start = body.index(pct)
    return write_claim(
        session,
        ClaimInput(
            subject_id=subj.id,
            predicate=Predicate.OWNS,
            object_id=obj.id,
            extractor="test@0",
            qualifiers={"stake_pct": float(pct.rstrip("%"))},
            valid_from=vfrom,
            evidence=[EvidenceInput(doc.id, start, start + len(pct), pct)],
        ),
    )


def test_supersession_closes_intervals_and_cleans_current_view(clean_session):
    session = clean_session
    body = "Filer reported 5.2% on 2024-03-01 and later 7.8% on 2024-06-01 and 9.1% on 2024-09-01."
    doc = _doc(session, body)
    filer = _ent(session, "Acme LP", "111")
    issuer = _ent(session, "Target Inc", "222")
    c1 = _owns(session, doc, filer, issuer, "5.2%", date(2024, 3, 1))
    c2 = _owns(session, doc, filer, issuer, "7.8%", date(2024, 6, 1))
    c3 = _owns(session, doc, filer, issuer, "9.1%", date(2024, 9, 1))
    session.flush()

    result = reconcile_supersessions(session)
    assert result["claims_superseded"] == 2

    session.refresh(c1)
    session.refresh(c2)
    session.refresh(c3)
    # Chain links and closed intervals.
    assert c1.superseded_by == c2.id and c1.valid_to == date(2024, 6, 1)
    assert c2.superseded_by == c3.id and c2.valid_to == date(2024, 9, 1)
    assert c3.superseded_by is None and c3.valid_to is None

    # Current world snapshot (valid today) = only the latest stake.
    current = claims_asof(session, valid_on=date.today(), subject_id=filer.id)
    assert [c.id for c in current] == [c3.id]

    # Mid-history valid-time snapshot returns the stake valid then.
    mid = claims_asof(session, valid_on=date(2024, 7, 1), subject_id=filer.id)
    assert [c.id for c in mid] == [c2.id]

    # Full timeline still shows all three stakes.
    tl = timeline(session, subject_id=filer.id, predicate=Predicate.OWNS)
    assert [c.qualifiers["stake_pct"] for c in tl] == [5.2, 7.8, 9.1]


def test_supersession_resolves_false_contradictions(clean_session):
    session = clean_session
    body = "Stake was 5.2% on 2024-03-01 then 7.8% on 2024-06-01 per amendment."
    doc = _doc(session, body)
    filer = _ent(session, "Beta LP", "333")
    issuer = _ent(session, "Omega Inc", "444")
    _owns(session, doc, filer, issuer, "5.2%", date(2024, 3, 1))
    _owns(session, doc, filer, issuer, "7.8%", date(2024, 6, 1))
    session.flush()

    # Before reconciliation: overlapping open intervals -> a contradiction.
    assert len(detect_contradictions(session)) >= 1
    reconcile_supersessions(session)
    # After: adjacent half-open intervals, earlier superseded -> no contradiction.
    assert detect_contradictions(session) == []


def test_materialize_real_contradiction_is_idempotent(clean_session):
    session = clean_session
    body = "Source A says 5.1% and source B says 9.2% for the same 2024 period."
    doc = _doc(session, body)
    filer = _ent(session, "Gamma LP", "555")
    issuer = _ent(session, "Sigma Inc", "666")
    # Same valid_from => genuine same-period conflict (not a supersession).
    _owns(session, doc, filer, issuer, "5.1%", date(2024, 1, 1))
    _owns(session, doc, filer, issuer, "9.2%", date(2024, 1, 1))
    session.flush()

    reconcile_supersessions(session)  # same date -> no supersession
    r1 = materialize_contradictions(session)
    assert r1["inserted"] >= 1
    n_after_first = len(session.execute(select(Contradiction)).scalars().all())
    # Re-run is idempotent (no duplicate rows).
    r2 = materialize_contradictions(session)
    assert r2["inserted"] == 0
    rows2 = session.execute(select(Contradiction)).scalars().all()
    assert len(rows2) == n_after_first
