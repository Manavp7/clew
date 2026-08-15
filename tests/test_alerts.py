"""Integration tests for watchlists + alert generation (idempotent, isolated)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select

from clew.alerts.service import run_alerts
from clew.db.models import Alert, Document, Source, Watch
from clew.db.repo import upsert_entity_by_external_id
from clew.ledger.writer import ClaimInput, EvidenceInput, write_claim
from clew.packs.pack_a_financial import EntityType, Predicate
from tests.conftest import requires_db

pytestmark = requires_db


def _setup_claim(session, stake):
    src = Source(name="T")
    session.add(src)
    session.flush()
    body = f"Holder owns {stake}% of Issuer."
    doc = Document(
        source_id=src.id,
        retrieved_at=datetime.now(UTC),
        content_hash=f"h{datetime.now(UTC).timestamp()}",
        text=body,
    )
    session.add(doc)
    session.flush()
    holder = upsert_entity_by_external_id(
        session, entity_type=EntityType.ORGANIZATION, canonical_name="Holder LP"
    )
    issuer = upsert_entity_by_external_id(
        session, entity_type=EntityType.ORGANIZATION, canonical_name="Issuer Inc"
    )
    s = body.index(f"{stake}%")
    write_claim(
        session,
        ClaimInput(
            subject_id=holder.id, predicate=Predicate.OWNS, object_id=issuer.id,
            extractor="t@0", qualifiers={"stake_pct": float(stake)}, valid_from=date(2024, 1, 1),
            evidence=[EvidenceInput(doc.id, s, s + len(f"{stake}%"), f"{stake}%")],
        ),
    )
    return issuer


def test_stake_threshold_alert_fires_and_dedupes(clean_session):
    session = clean_session
    issuer = _setup_claim(session, "12.5")
    session.add(Watch(kind="stake_threshold", target=issuer.id, threshold=10.0))
    session.flush()

    # Pass the transactional session so the run stays isolated (rolled back at teardown).
    assert run_alerts(session)["alerts_fired"] == 1
    assert run_alerts(session)["alerts_fired"] == 0  # idempotent dedupe

    alerts = session.execute(select(Alert)).scalars().all()
    assert len(alerts) == 1 and "12.5%" in alerts[0].message


def test_below_threshold_does_not_fire(clean_session):
    session = clean_session
    issuer = _setup_claim(session, "3.0")
    session.add(Watch(kind="stake_threshold", target=issuer.id, threshold=10.0))
    session.flush()
    assert run_alerts(session)["alerts_fired"] == 0
