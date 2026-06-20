"""Resolution stage: mentions + filing headers -> canonical entities.

Builds resolution records from (a) all Organization/Person mentions and (b) the
CIK-anchored filer/subject records in each filing header, clusters them, then
creates canonical entities and back-fills ``mention.resolved_to``.
"""

from __future__ import annotations

from sqlalchemy import select

from clew.config import get_settings
from clew.db.models import Document, Entity, Mention
from clew.db.repo import upsert_entity_by_external_id
from clew.db.session import write_session
from clew.packs.pack_a_financial import EntityType
from clew.resolve.normalize import core
from clew.resolve.resolver import Cluster, Record, resolve_records

_TYPE_FROM_NER = {"Organization": EntityType.ORGANIZATION, "Person": EntityType.PERSON}


def _collect_records(session) -> tuple[list[Record], dict[int, int]]:
    """Return (records, rid->mention_id) from mentions + filing headers."""
    records: list[Record] = []
    rid = 0

    # (a) Org/Person mentions.
    mentions = session.execute(
        select(Mention).where(Mention.ner_type.in_(["Organization", "Person"]))
    ).scalars().all()
    for m in mentions:
        records.append(
            Record(
                rid=rid,
                name=m.surface_text,
                entity_type=_TYPE_FROM_NER.get(m.ner_type, EntityType.ORGANIZATION),
                cik=None,
                mention_id=m.id,
            )
        )
        rid += 1

    # (b) Header filer/subject records (CIK anchors).
    docs = session.execute(select(Document)).scalars().all()
    for d in docs:
        meta = d.meta or {}
        for f in meta.get("filers", []):
            if f.get("name"):
                records.append(
                    Record(rid=rid, name=f["name"], entity_type=_guess_type(f["name"]),
                           cik=f.get("cik"))
                )
                rid += 1
        for s in meta.get("subject_companies", []):
            if s.get("name"):
                records.append(
                    Record(rid=rid, name=s["name"], entity_type=EntityType.ORGANIZATION,
                           cik=s.get("cik"))
                )
                rid += 1

    return records, {}


def _guess_type(name: str) -> str:
    # Filers can be persons or orgs; a light heuristic (org keywords) else Person.
    low = name.lower()
    org_kw = ("llc", "l.p.", "lp", "inc", "ltd", "corp", "partners", "capital",
              "management", "fund", "trust", "group", "holdings", "company", "plc")
    return EntityType.ORGANIZATION if any(k in low for k in org_kw) else EntityType.PERSON


def _build_name_index(session) -> dict[tuple[str, str], Entity]:
    """Index existing entities by (type, core-name) for O(1) reuse during resolution."""
    index: dict[tuple[str, str], Entity] = {}
    for e in session.execute(select(Entity)).scalars().all():
        for name in [e.canonical_name, *(e.aliases or [])]:
            index.setdefault((e.type, core(name)), e)
    return index


def _find_by_name(session, entity_type: str, names: list[str]) -> Entity | None:
    """Reuse an existing entity whose canonical_name/alias matches by core name."""
    targets = {core(n) for n in names}
    candidates = session.execute(
        select(Entity).where(Entity.type == entity_type)
    ).scalars().all()
    for e in candidates:
        pool = [e.canonical_name, *(e.aliases or [])]
        if any(core(p) in targets for p in pool):
            return e
    return None


def _materialize_cluster(
    session, cluster: Cluster, index: dict[tuple[str, str], Entity] | None = None
) -> Entity:
    cik = cluster.cik
    if cik:
        entity = upsert_entity_by_external_id(
            session,
            entity_type=cluster.entity_type,
            canonical_name=cluster.canonical_name,
            namespace="cik",
            value=cik,
            aliases=cluster.aliases,
        )
    else:
        # Non-CIK cluster: reuse an existing same-name entity for idempotency.
        names = [cluster.canonical_name, *cluster.aliases]
        existing = None
        if index is not None:
            for n in names:
                existing = index.get((cluster.entity_type, core(n)))
                if existing is not None:
                    break
        else:
            existing = _find_by_name(session, cluster.entity_type, names)
        if existing is not None:
            merged = set(existing.aliases or []) | set(cluster.aliases)
            merged.discard(existing.canonical_name)
            existing.aliases = sorted(merged)
            entity = existing
        else:
            entity = upsert_entity_by_external_id(
                session,
                entity_type=cluster.entity_type,
                canonical_name=cluster.canonical_name,
                aliases=cluster.aliases,
            )
    # Keep the index current so later clusters in the same run can reuse this entity.
    if index is not None:
        for name in [entity.canonical_name, *(entity.aliases or [])]:
            index.setdefault((entity.type, core(name)), entity)
    return entity


def run_resolution(backend: str | None = None) -> dict:
    backend = backend or "default"
    with write_session() as session:
        records, _ = _collect_records(session)
        if backend == "splink":
            from clew.resolve.splink_er import resolve_records_splink

            clusters = resolve_records_splink(records)
        else:
            clusters = resolve_records(records)

        index = _build_name_index(session)
        entities_created = 0
        mentions_resolved = 0
        for cluster in clusters:
            entity = _materialize_cluster(session, cluster, index)
            entities_created += 1
            for member in cluster.members:
                if member.mention_id is not None:
                    m = session.get(Mention, member.mention_id)
                    if m is not None:
                        m.resolved_to = entity.id
                        m.resolution_confidence = cluster.confidence
                        mentions_resolved += 1

    return {
        "records": len(records),
        "clusters": len(clusters),
        "mentions_resolved": mentions_resolved,
        "backend": backend,
    }


def link_or_create_entity(
    session, *, surface: str, entity_type: str, cik: str | None = None
) -> Entity:
    """Resolve a claim's subject/object surface to a canonical entity.

    Prefers CIK anchor, then name match against existing entities, else creates a
    new (singleton) canonical entity. Keeps the ledger's entities authoritative
    while letting extraction reference them.
    """
    if cik:
        return upsert_entity_by_external_id(
            session, entity_type=entity_type, canonical_name=surface,
            namespace="cik", value=cik, aliases=[surface],
        )
    existing = _find_by_name(session, entity_type, [surface])
    if existing is not None:
        return existing
    return upsert_entity_by_external_id(
        session, entity_type=entity_type, canonical_name=surface, aliases=[]
    )


def get_settings_backend() -> str:
    # placeholder hook if we later move backend selection into config
    return "default" if get_settings() else "default"
