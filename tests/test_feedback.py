"""Tests for the ER continuous-learning loop (merge op + suggestions + feedback)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from clew.db.models import Entity, MergeSuggestion
from clew.db.repo import upsert_entity_by_external_id
from clew.ledger.writer import ClaimInput, EvidenceInput, write_claim
from clew.packs.pack_a_financial import EntityType, Predicate
from clew.resolve.feedback import (
    accept_suggestion,
    generate_suggestions,
    load_decisions,
    reject_suggestion,
)
from clew.resolve.merge import merge_entities
from clew.resolve.normalize import core
from clew.resolve.resolver import Record, resolve_records
from tests.conftest import requires_db

pytestmark = requires_db


def _org(session, name):
    return upsert_entity_by_external_id(
        session, entity_type=EntityType.ORGANIZATION, canonical_name=name
    )


def _doc_with_claim(session, keep, drop):
    from clew.db.models import Document, Source

    src = Source(name="T")
    session.add(src)
    session.flush()
    doc = Document(
        source_id=src.id,
        retrieved_at=datetime.now(UTC),
        content_hash=f"h{datetime.now(UTC).timestamp()}",
        text="Drop Co owns 5% of something.",
    )
    session.add(doc)
    session.flush()
    write_claim(
        session,
        ClaimInput(
            subject_id=drop.id,
            predicate=Predicate.OWNS,
            object_id=keep.id,
            extractor="t@0",
            evidence=[EvidenceInput(doc.id, 0, 7, "Drop Co")],
        ),
    )
    return doc


def test_merge_entities_repoints_refs(clean_session):
    session = clean_session
    keep = _org(session, "Vanguard Group Inc")
    drop = _org(session, "Vanguard Group, Inc.")
    obj = _org(session, "Some Issuer Inc")
    _doc_with_claim(session, obj, drop)  # drop is subject of a claim
    session.flush()

    result = merge_entities(session, keep.id, drop.id)
    assert result["claims_repointed"] == 1
    # dropped entity gone; its name absorbed as an alias of keep.
    assert session.get(Entity, drop.id) is None
    refreshed = session.get(Entity, keep.id)
    assert "Vanguard Group, Inc." in refreshed.aliases


def test_suggest_accept_creates_must_link_and_merges(clean_session):
    session = clean_session
    # Two grey-zone names (JW ~0.90, below the 0.92 auto-merge cutoff).
    a = _org(session, "Stonepeak Advisors")
    b = _org(session, "Stonepoint Advisors")
    session.flush()

    res = generate_suggestions(session)
    assert res["suggested"] >= 1
    sug = session.execute(select(MergeSuggestion)).scalars().first()
    out = accept_suggestion(session, sug.id)
    assert out["kept"] in (a.id, b.id)
    # A must_link decision is recorded and surfaced to the resolver.
    must_link, _ = load_decisions(session)
    assert frozenset({core(a.canonical_name), core(b.canonical_name)}) in must_link


def test_reject_blocks_future_merge(clean_session):
    session = clean_session
    a = _org(session, "Birchwood Capital")
    b = _org(session, "Birchwald Capital")
    session.flush()
    generate_suggestions(session)
    sug = session.execute(select(MergeSuggestion)).scalars().first()
    reject_suggestion(session, sug.id)
    _, must_not = load_decisions(session)
    key = frozenset({core(a.canonical_name), core(b.canonical_name)})
    assert key in must_not
    # The resolver must NOT merge a must-not-link pair even if names are similar.
    recs = [
        Record(0, a.canonical_name, EntityType.ORGANIZATION),
        Record(1, b.canonical_name, EntityType.ORGANIZATION),
    ]
    clusters = resolve_records(recs, must_not_link=must_not)
    assert len(clusters) == 2
