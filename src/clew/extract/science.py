"""Pack-B extraction: OpenAlex works -> nanopublication-style claims.

Reuses the domain-agnostic core (ledger writer, entity repo, projections). Each
claim is grounded to a span in the work's rendered text and attributed to an
extractor version; entities resolve deterministically via OpenAlex/DOI/ORCID/ROR
external-id anchors.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from clew.db.models import Document, Evidence
from clew.db.repo import upsert_entity_by_external_id
from clew.db.session import write_session
from clew.extract.grounding import ground_quote
from clew.ledger.writer import ClaimInput, EvidenceInput, write_claim
from clew.packs.pack_b_science import EntityType, Predicate

EXTRACTOR = "science-meta@0.1"


def _pub_date(meta: dict) -> date | None:
    pd = meta.get("publication_date")
    if not pd:
        return None
    try:
        return date.fromisoformat(pd)
    except ValueError:
        return None


def _ground(session, doc, quote: str):
    g = ground_quote(doc.text, quote)
    if g is None:
        return None
    start, end, snippet = g
    return EvidenceInput(doc.id, start, end, snippet, "metadata", EXTRACTOR)


def _paper_entity(session, meta: dict):
    return upsert_entity_by_external_id(
        session,
        entity_type=EntityType.PAPER,
        canonical_name=meta.get("title") or meta["openalex"],
        namespace="openalex",
        value=meta["openalex"],
        external_ids={"doi": meta["doi"]} if meta.get("doi") else None,
    )


def _process_doc(session, doc: Document) -> int:
    meta = doc.meta or {}
    if not meta.get("openalex"):
        return 0
    valid_from = _pub_date(meta)
    paper = _paper_entity(session, meta)
    written = 0

    # AUTHORED + AFFILIATED_WITH
    for a in meta.get("authors", []):
        if not a.get("name"):
            continue
        author = upsert_entity_by_external_id(
            session,
            entity_type=EntityType.AUTHOR,
            canonical_name=a["name"],
            namespace="openalex_author" if a.get("id") else None,
            value=a.get("id"),
            external_ids={"orcid": a["orcid"]} if a.get("orcid") else None,
        )
        ev = _ground(session, doc, a["name"])
        if ev is not None:
            write_claim(
                session,
                ClaimInput(
                    subject_id=author.id, predicate=Predicate.AUTHORED, object_id=paper.id,
                    extractor=EXTRACTOR, valid_from=valid_from, evidence=[ev],
                ),
            )
            written += 1
        for inst in a.get("institutions", []):
            if not inst.get("name"):
                continue
            institution = upsert_entity_by_external_id(
                session,
                entity_type=EntityType.INSTITUTION,
                canonical_name=inst["name"],
                namespace="openalex_institution" if inst.get("id") else None,
                value=inst.get("id"),
                external_ids={"ror": inst["ror"]} if inst.get("ror") else None,
            )
            ev = _ground(session, doc, f"{a['name']} — {inst['name']}")
            if ev is not None:
                write_claim(
                    session,
                    ClaimInput(
                        subject_id=author.id, predicate=Predicate.AFFILIATED_WITH,
                        object_id=institution.id, extractor=EXTRACTOR, valid_from=valid_from,
                        evidence=[ev],
                    ),
                )
                written += 1

    # MENTIONS_CONCEPT
    for c in meta.get("concepts", []):
        if not c.get("name") or not c.get("id"):
            continue
        concept = upsert_entity_by_external_id(
            session, entity_type=EntityType.CONCEPT, canonical_name=c["name"],
            namespace="openalex_concept", value=c["id"],
        )
        ev = _ground(session, doc, c["name"])
        if ev is not None:
            write_claim(
                session,
                ClaimInput(
                    subject_id=paper.id, predicate=Predicate.MENTIONS_CONCEPT,
                    object_id=concept.id, extractor=EXTRACTOR, valid_from=valid_from, evidence=[ev],
                ),
            )
            written += 1

    # CITES (reference paper stubs anchored by OpenAlex id)
    for ref in meta.get("references", []):
        if not ref:
            continue
        ref_paper = upsert_entity_by_external_id(
            session, entity_type=EntityType.PAPER, canonical_name=ref,
            namespace="openalex", value=ref,
        )
        ev = _ground(session, doc, ref)
        if ev is not None:
            write_claim(
                session,
                ClaimInput(
                    subject_id=paper.id, predicate=Predicate.CITES, object_id=ref_paper.id,
                    extractor=EXTRACTOR, valid_from=valid_from, evidence=[ev],
                ),
            )
            written += 1

    return written


def run_science_claims(limit: int | None = None) -> dict:
    written, docs_processed = 0, 0
    with write_session() as session:
        done = set(
            session.execute(select(Evidence.document_id).distinct()).scalars().all()
        )
        docs = session.execute(
            select(Document).where(Document.doc_type == "openalex_work")
        ).scalars().all()
        if limit:
            docs = docs[:limit]
        for doc in docs:
            if doc.id in done:
                continue
            docs_processed += 1
            written += _process_doc(session, doc)
    return {"docs_processed": docs_processed, "claims_written": written}
