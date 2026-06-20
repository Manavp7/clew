"""Repository helpers: entity id allocation, lookup, and canonical upsert.

Entities get a typed, padded id (``ORG_000123`` / ``PER_000456`` / ``SEC_...``)
from a single Postgres sequence. The type prefix encodes :class:`EntityType`.
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from clew.db.models import Entity
from clew.packs.pack_a_financial import EntityType

# Typed entity-id prefixes across all packs (the type prefix encodes the type).
_TYPE_PREFIX: dict[str, str] = {
    EntityType.ORGANIZATION: "ORG",
    EntityType.PERSON: "PER",
    EntityType.SECURITY: "SEC",
    # Pack B (scientific)
    "Paper": "PAP",
    "Author": "AUT",
    "Institution": "INS",
    "Concept": "CON",
}


def _prefix_for(entity_type: str) -> str:
    return _TYPE_PREFIX.get(entity_type, "ENT")


def allocate_entity_id(session: Session, entity_type: str) -> str:
    """Allocate a globally-unique, type-prefixed entity id."""
    n = session.execute(text("SELECT nextval('entity_id_seq')")).scalar_one()
    return f"{_prefix_for(entity_type)}_{n:06d}"


def find_entity_by_external_id(session: Session, namespace: str, value: str) -> Entity | None:
    """Find a canonical entity by an external id anchor (e.g. cik, lei)."""
    stmt = select(Entity).where(Entity.external_ids[namespace].astext == str(value))
    return session.execute(stmt).scalars().first()


def get_entity(session: Session, entity_id: str) -> Entity | None:
    return session.get(Entity, entity_id)


def upsert_entity_by_external_id(
    session: Session,
    *,
    entity_type: str,
    canonical_name: str,
    namespace: str | None = None,
    value: str | None = None,
    aliases: list[str] | None = None,
    external_ids: dict | None = None,
) -> Entity:
    """Create or fetch a canonical entity, anchored on an external id when given.

    External-id anchors (CIK/LEI/CUSIP) give deterministic, high-precision
    resolution and are the backbone that keeps ER from poisoning the graph.
    """
    existing: Entity | None = None
    if namespace and value:
        existing = find_entity_by_external_id(session, namespace, value)

    if existing is not None:
        _merge_aliases(existing, aliases)
        if external_ids:
            existing.external_ids = {**(existing.external_ids or {}), **external_ids}
        return existing

    ext = dict(external_ids or {})
    if namespace and value:
        ext.setdefault(namespace, str(value))

    entity = Entity(
        id=allocate_entity_id(session, entity_type),
        type=entity_type,
        canonical_name=canonical_name,
        aliases=sorted(set(aliases or [])),
        external_ids=ext,
    )
    session.add(entity)
    session.flush()
    return entity


def _merge_aliases(entity: Entity, aliases: list[str] | None) -> None:
    if not aliases:
        return
    merged = set(entity.aliases or []) | set(aliases)
    merged.discard(entity.canonical_name)
    entity.aliases = sorted(merged)
