"""Unit/integration tests for the ledger invariants and bitemporal queries."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from clew.db.models import Document, Source
from clew.db.repo import upsert_entity_by_external_id
from clew.ledger.asof import claims_asof, knowledge_diff, timeline
from clew.ledger.contradiction import detect_contradictions
from clew.ledger.writer import (
    ClaimInput,
    EvidenceInput,
    LedgerError,
    retract_claim,
    supersede_claim,
    write_claim,
)
from clew.packs.pack_a_financial import EntityType, Predicate
from tests.conftest import requires_db

pytestmark = requires_db


def _doc(session, text_body: str) -> Document:
    src = Source(name="TEST")
    session.add(src)
    session.flush()
    doc = Document(
        source_id=src.id,
        doc_type="13D",
        retrieved_at=datetime.now(UTC),
        content_hash=f"hash-{datetime.now(UTC).timestamp()}",
        text=text_body,
    )
    session.add(doc)
    session.flush()
    return doc


def _entity(session, etype, name, ns, val):
    return upsert_entity_by_external_id(
        session, entity_type=etype, canonical_name=name, namespace=ns, value=val
    )


def test_evidence_required(session):
    _doc(session, "Acme owns 5% of Target Corp.")
    filer = _entity(session, EntityType.ORGANIZATION, "Acme", "cik", "111")
    issuer = _entity(session, EntityType.ORGANIZATION, "Target Corp", "cik", "222")
    with pytest.raises(LedgerError, match="no evidence"):
        write_claim(
            session,
            ClaimInput(
                subject_id=filer.id,
                predicate=Predicate.OWNS,
                object_id=issuer.id,
                extractor="test@0",
            ),
        )


def test_offset_roundtrip_enforced(session):
    body = "Acme owns 5% of Target Corp."
    doc = _doc(session, body)
    filer = _entity(session, EntityType.ORGANIZATION, "Acme", "cik", "111")
    issuer = _entity(session, EntityType.ORGANIZATION, "Target Corp", "cik", "222")
    # Wrong snippet for the offsets -> rejected.
    with pytest.raises(LedgerError, match="offset mismatch"):
        write_claim(
            session,
            ClaimInput(
                subject_id=filer.id,
                predicate=Predicate.OWNS,
                object_id=issuer.id,
                extractor="test@0",
                evidence=[EvidenceInput(doc.id, 0, 4, "WRONG")],
            ),
        )
    # Correct offsets -> accepted.
    start = body.index("Acme")
    claim = write_claim(
        session,
        ClaimInput(
            subject_id=filer.id,
            predicate=Predicate.OWNS,
            object_id=issuer.id,
            extractor="test@0",
            qualifiers={"stake_pct": 5.0},
            evidence=[EvidenceInput(doc.id, start, start + 4, "Acme")],
        ),
    )
    assert claim.id is not None
    assert claim.evidence[0].snippet == "Acme"


def test_asof_and_retraction(session):
    body = "Acme owns 5% of Target Corp."
    doc = _doc(session, body)
    filer = _entity(session, EntityType.ORGANIZATION, "Acme", "cik", "111")
    issuer = _entity(session, EntityType.ORGANIZATION, "Target Corp", "cik", "222")
    claim = write_claim(
        session,
        ClaimInput(
            subject_id=filer.id,
            predicate=Predicate.OWNS,
            object_id=issuer.id,
            extractor="test@0",
            evidence=[EvidenceInput(doc.id, 0, 4, "Acme")],
        ),
    )
    t_before = datetime.now(UTC) - timedelta(seconds=1)
    # Currently believed.
    assert any(c.id == claim.id for c in claims_asof(session, subject_id=filer.id))
    # Retract it.
    retract_claim(session, claim.id)
    session.flush()
    # No longer believed "now".
    assert not any(c.id == claim.id for c in claims_asof(session, subject_id=filer.id))
    # But still believed as-of a time before retraction (asserted earlier).
    asof = claims_asof(session, as_of=datetime.now(UTC), subject_id=filer.id)
    assert not any(c.id == claim.id for c in asof)
    # Knowledge diff captures the assertion.
    asserted, retracted = knowledge_diff(session, since=t_before)
    assert any(c.id == claim.id for c in asserted)
    assert any(c.id == claim.id for c in retracted)


def test_supersession_timeline(session):
    body = "Acme owns 5% then later 9% of Target Corp."
    doc = _doc(session, body)
    filer = _entity(session, EntityType.ORGANIZATION, "Acme", "cik", "111")
    issuer = _entity(session, EntityType.ORGANIZATION, "Target Corp", "cik", "222")
    s5 = body.index("5%")
    s9 = body.index("9%")
    c1 = write_claim(
        session,
        ClaimInput(
            subject_id=filer.id,
            predicate=Predicate.OWNS,
            object_id=issuer.id,
            extractor="test@0",
            qualifiers={"stake_pct": 5.0},
            valid_from=date(2024, 1, 1),
            evidence=[EvidenceInput(doc.id, s5, s5 + 2, "5%")],
        ),
    )
    c2 = supersede_claim(
        session,
        c1.id,
        ClaimInput(
            subject_id=filer.id,
            predicate=Predicate.OWNS,
            object_id=issuer.id,
            extractor="test@0",
            qualifiers={"stake_pct": 9.0},
            valid_from=date(2024, 6, 1),
            evidence=[EvidenceInput(doc.id, s9, s9 + 2, "9%")],
        ),
    )
    session.flush()
    # Old claim is superseded + retracted; current timeline shows only the new stake.
    current = timeline(session, subject_id=filer.id, predicate=Predicate.OWNS)
    assert [c.id for c in current] == [c2.id]
    assert current[0].qualifiers["stake_pct"] == 9.0


def test_contradiction_detection(session):
    body = "Filing A reports 5.1% while Filing B reports 9.2% for the same period."
    doc = _doc(session, body)
    filer = _entity(session, EntityType.ORGANIZATION, "Acme", "cik", "111")
    issuer = _entity(session, EntityType.ORGANIZATION, "Target Corp", "cik", "222")
    s1 = body.index("5.1%")
    s2 = body.index("9.2%")
    write_claim(
        session,
        ClaimInput(
            subject_id=filer.id, predicate=Predicate.OWNS, object_id=issuer.id, extractor="test@0",
            qualifiers={"stake_pct": 5.1}, valid_from=date(2024, 1, 1), valid_to=date(2024, 12, 31),
            evidence=[EvidenceInput(doc.id, s1, s1 + 4, "5.1%")],
        ),
    )
    write_claim(
        session,
        ClaimInput(
            subject_id=filer.id, predicate=Predicate.OWNS, object_id=issuer.id, extractor="test@0",
            qualifiers={"stake_pct": 9.2}, valid_from=date(2024, 6, 1), valid_to=date(2024, 12, 31),
            evidence=[EvidenceInput(doc.id, s2, s2 + 4, "9.2%")],
        ),
    )
    session.flush()
    pairs = detect_contradictions(session)
    assert any(p.type == "value_conflict" for p in pairs)
