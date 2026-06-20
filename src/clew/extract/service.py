"""Extraction stages: mention detection and claim writing.

* :func:`run_mentions` — GLiNER mentions for every document (offset-grounded).
* :func:`run_claims` — rule baseline (always) + optional LLM extraction, grounded
  to evidence offsets, resolved to canonical entities, written to the ledger.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from clew.config import get_settings
from clew.db.models import Document
from clew.db.session import write_session
from clew.extract.claims import LLMClaimExtractor, RuleClaimExtractor
from clew.extract.grounding import ground_quote
from clew.extract.ner import GLINER_VERSION, extract_mentions
from clew.extract.schemas import ExtractedClaim
from clew.ledger.writer import ClaimInput, EvidenceInput, LedgerError, write_claim
from clew.packs.pack_a_financial import EntityType, is_valid_predicate
from clew.parse.edgar import parse_filing
from clew.resolve.service import link_or_create_entity


def run_mentions(threshold: float = 0.5, limit: int | None = None) -> dict:
    written = 0
    with write_session() as session:
        from clew.db.models import Mention
        from clew.ledger.writer import write_mention

        done_doc_ids = set(
            session.execute(select(Mention.document_id).distinct()).scalars().all()
        )
        docs = session.execute(select(Document)).scalars().all()
        if limit:
            docs = docs[:limit]
        for doc in docs:
            if not doc.text or doc.id in done_doc_ids:
                continue  # incremental: skip already-processed documents
            spans = extract_mentions(doc.text, threshold=threshold)
            for s in spans:
                write_mention(
                    session,
                    document_id=doc.id,
                    surface_text=s.surface,
                    char_start=s.start,
                    char_end=s.end,
                    ner_type=s.ner_type,
                    extractor=GLINER_VERSION,
                )
                written += 1
    return {"mentions_written": written}


def _header_cik_lookup(meta: dict) -> dict[str, str]:
    """Map header entity names -> CIK for deterministic claim-entity linking."""
    out: dict[str, str] = {}
    for f in meta.get("filers", []):
        if f.get("name") and f.get("cik"):
            out[f["name"]] = f["cik"]
    for s in meta.get("subject_companies", []):
        if s.get("name") and s.get("cik"):
            out[s["name"]] = s["cik"]
    return out


def _entity_type_for(surface: str, predicate: str, is_object: bool) -> str:
    # Issuer (object of OWNS) is always an Organization.
    if is_object:
        return EntityType.ORGANIZATION
    low = surface.lower()
    org_kw = ("llc", "l.p.", "lp", "inc", "ltd", "corp", "partners", "capital",
              "management", "fund", "trust", "group", "holdings", "company", "plc")
    return EntityType.ORGANIZATION if any(k in low for k in org_kw) else EntityType.PERSON


def _to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _write_one_claim(session, doc, ec: ExtractedClaim, method: str, model_version: str) -> bool:
    if not is_valid_predicate(ec.predicate):
        return False
    grounded = ground_quote(doc.text, ec.evidence_quote)
    if grounded is None:
        return False
    start, end, snippet = grounded

    cik_lookup = _header_cik_lookup(doc.meta or {})
    subj = link_or_create_entity(
        session,
        surface=ec.subject_surface,
        entity_type=_entity_type_for(ec.subject_surface, ec.predicate, is_object=False),
        cik=cik_lookup.get(ec.subject_surface),
    )
    obj_id = None
    if ec.object_surface:
        obj = link_or_create_entity(
            session,
            surface=ec.object_surface,
            entity_type=_entity_type_for(ec.object_surface, ec.predicate, is_object=True),
            cik=cik_lookup.get(ec.object_surface),
        )
        obj_id = obj.id

    try:
        write_claim(
            session,
            ClaimInput(
                subject_id=subj.id,
                predicate=ec.predicate,
                object_id=obj_id,
                object_literal=ec.object_literal,
                qualifiers=ec.qualifiers or {},
                valid_from=_to_date(ec.valid_from),
                polarity=ec.polarity,
                confidence=ec.confidence,
                extractor=model_version,
                evidence=[EvidenceInput(doc.id, start, end, snippet, method, model_version)],
            ),
        )
        return True
    except LedgerError:
        return False


def run_claims(use_llm: bool | None = None, limit: int | None = None) -> dict:
    settings = get_settings()
    if use_llm is None:
        use_llm = settings.has_llm

    rule = RuleClaimExtractor()
    llm = LLMClaimExtractor() if use_llm else None

    written_rule, written_llm, docs_processed = 0, 0, 0
    with write_session() as session:
        from clew.db.models import Evidence

        done_doc_ids = set(
            session.execute(select(Evidence.document_id).distinct()).scalars().all()
        )
        docs = session.execute(select(Document)).scalars().all()
        if limit:
            docs = docs[:limit]
        for doc in docs:
            if not doc.text or doc.id in done_doc_ids:
                continue  # incremental: skip documents that already have claims
            docs_processed += 1
            parsed = parse_filing(doc.text)
            meta = doc.meta or {}

            for ec in rule.extract(text=doc.text, parsed=parsed, meta=meta):
                if _write_one_claim(session, doc, ec, rule.method, rule.version):
                    written_rule += 1

            if llm is not None:
                try:
                    ecs = llm.extract(text=doc.text, parsed=parsed, meta=meta)
                except Exception as exc:  # noqa: BLE001 - LLM failures shouldn't abort the run
                    print(f"  ! LLM extract failed for doc {doc.id}: {exc}")
                    ecs = []
                for ec in ecs:
                    if _write_one_claim(session, doc, ec, llm.method, llm.version):
                        written_llm += 1

    return {
        "docs_processed": docs_processed,
        "claims_rule": written_rule,
        "claims_llm": written_llm,
        "llm_used": bool(llm),
    }
