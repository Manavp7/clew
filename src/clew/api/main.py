"""Read-only API over the ledger + projections.

Every endpoint uses :func:`clew.db.session.read_session` (a ``READ ONLY``
transaction), enforcing the constraint that the reasoning/serving layer cannot
write claims. Serves a minimal Sigma.js UI at ``/``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from clew.db.models import Claim, Document, Entity, Evidence
from clew.db.session import read_session
from clew.ledger.asof import claims_asof, timeline
from clew.project.graph import build_graph, ego_graph, graph_to_json

app = FastAPI(title="clew — Canonical Intelligence Engine", version="0.1.0")

_WEB = Path(__file__).resolve().parents[3] / "web"


def _parse_asof(as_of: str | None) -> datetime | None:
    if not as_of:
        return None
    return datetime.fromisoformat(as_of)


def _claim_dict(session, c: Claim) -> dict:
    subj = session.get(Entity, c.subject_id)
    obj = session.get(Entity, c.object_id) if c.object_id else None
    ev = session.execute(select(Evidence).where(Evidence.claim_id == c.id)).scalars().all()
    return {
        "id": c.id,
        "subject": {"id": subj.id, "name": subj.canonical_name} if subj else None,
        "predicate": c.predicate,
        "object": {"id": obj.id, "name": obj.canonical_name} if obj else None,
        "object_literal": c.object_literal,
        "qualifiers": c.qualifiers,
        "valid_from": c.valid_from.isoformat() if c.valid_from else None,
        "valid_to": c.valid_to.isoformat() if c.valid_to else None,
        "asserted_at": c.asserted_at.isoformat() if c.asserted_at else None,
        "retracted_at": c.retracted_at.isoformat() if c.retracted_at else None,
        "confidence": c.confidence,
        "extractor": c.extractor,
        "evidence": [
            {
                "document_id": e.document_id,
                "char_start": e.char_start,
                "char_end": e.char_end,
                "snippet": e.snippet,
                "method": e.method,
            }
            for e in ev
        ],
    }


@app.get("/health")
def health() -> dict:
    with read_session() as session:
        n_claims = session.execute(select(Claim.id)).scalars().all()
        n_entities = session.execute(select(Entity.id)).scalars().all()
    return {"status": "ok", "claims": len(n_claims), "entities": len(n_entities)}


@app.get("/entities/{entity_id}")
def get_entity(entity_id: str) -> dict:
    with read_session() as session:
        e = session.get(Entity, entity_id)
        if e is None:
            raise HTTPException(404, f"entity {entity_id} not found")
        as_subject = claims_asof(session, subject_id=entity_id)
        as_object = session.execute(
            select(Claim).where(Claim.object_id == entity_id, Claim.retracted_at.is_(None))
        ).scalars().all()
        return {
            "id": e.id,
            "type": e.type,
            "canonical_name": e.canonical_name,
            "aliases": e.aliases,
            "external_ids": e.external_ids,
            "claims_as_subject": [_claim_dict(session, c) for c in as_subject],
            "claims_as_object": [_claim_dict(session, c) for c in as_object],
        }


@app.get("/claims")
def list_claims(
    subject: str | None = None,
    predicate: str | None = None,
    object_id: str | None = Query(default=None, alias="object"),
    as_of: str | None = None,
) -> dict:
    with read_session() as session:
        claims = claims_asof(
            session,
            as_of=_parse_asof(as_of),
            subject_id=subject,
            predicate=predicate,
            object_id=object_id,
        )
        return {"count": len(claims), "claims": [_claim_dict(session, c) for c in claims]}


@app.get("/timeline")
def get_timeline(
    subject: str,
    predicate: str = "OWNS",
    object_id: str | None = Query(default=None, alias="object"),
    as_of: str | None = None,
) -> dict:
    """Stake-change timeline for a (subject, predicate[, object]) relationship."""
    with read_session() as session:
        claims = timeline(
            session,
            subject_id=subject,
            predicate=predicate,
            object_id=object_id,
            as_of=_parse_asof(as_of),
        )
        points = [
            {
                "claim_id": c.id,
                "valid_from": c.valid_from.isoformat() if c.valid_from else None,
                "valid_to": c.valid_to.isoformat() if c.valid_to else None,
                "stake_pct": (c.qualifiers or {}).get("stake_pct"),
                "object": c.object_id,
                "evidence": [
                    {"snippet": e.snippet, "document_id": e.document_id}
                    for e in session.execute(
                        select(Evidence).where(Evidence.claim_id == c.id)
                    ).scalars().all()
                ],
            }
            for c in claims
        ]
        return {"subject": subject, "predicate": predicate, "points": points}


@app.get("/search")
def search(q: str, limit: int = 10) -> dict:
    from clew.project.vectors import search_entities

    return {"query": q, "results": search_entities(q, limit=limit)}


@app.get("/graph")
def get_graph(
    center: str | None = None, radius: int = 1, as_of: str | None = None
) -> dict:
    with read_session() as session:
        g = build_graph(session, as_of=_parse_asof(as_of))
        if center:
            g = ego_graph(g, center, radius=radius)
        return graph_to_json(g)


@app.get("/documents/{document_id}")
def get_document(document_id: int) -> dict:
    with read_session() as session:
        d = session.get(Document, document_id)
        if d is None:
            raise HTTPException(404, f"document {document_id} not found")
        return {
            "id": d.id,
            "doc_type": d.doc_type,
            "external_id": d.external_id,
            "url": d.url,
            "text": d.text,
        }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = _WEB / "index.html"
    if html.exists():
        return html.read_text()
    return "<h1>clew</h1><p>UI not found.</p>"
