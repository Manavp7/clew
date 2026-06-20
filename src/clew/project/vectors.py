"""Vector projection — entity embeddings into pgvector (rebuildable).

Embeds each canonical entity's name + aliases with the configured embedding
model (Qwen3-Embedding on CPU by default) and stores the vector on
``entity.embedding`` for semantic search. Like the graph, this is a projection
of the ledger and can be rebuilt at will.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Entity
from clew.db.session import write_session


def _entity_text(e: Entity) -> str:
    parts = [e.canonical_name, *(e.aliases or [])]
    return " | ".join(dict.fromkeys(p for p in parts if p))


def build_entity_vectors(batch_size: int = 64, only_missing: bool = False) -> dict:
    from clew.llm.gateway import get_embedder

    embedder = get_embedder()
    embedded = 0
    with write_session() as session:
        stmt = select(Entity)
        if only_missing:
            stmt = stmt.where(Entity.embedding.is_(None))
        entities = session.execute(stmt).scalars().all()
        for i in range(0, len(entities), batch_size):
            chunk = entities[i : i + batch_size]
            vectors = embedder.embed([_entity_text(e) for e in chunk])
            for e, vec in zip(chunk, vectors, strict=True):
                e.embedding = vec
                embedded += 1
    return {"entities_embedded": embedded, "model": embedder.model, "dim": embedder.dim}


def search_entities(query: str, *, limit: int = 10) -> list[dict]:
    """Semantic nearest-neighbour search over entity embeddings (pgvector)."""
    from clew.db.session import read_session
    from clew.llm.gateway import get_embedder

    embedder = get_embedder()
    qvec = embedder.embed([query])[0]
    with read_session() as session:
        rows = _knn(session, qvec, limit)
    return rows


def _knn(session: Session, qvec: list[float], limit: int) -> list[dict]:
    # pgvector cosine distance operator <=>
    stmt = (
        select(
            Entity.id,
            Entity.canonical_name,
            Entity.type,
            Entity.embedding.cosine_distance(qvec).label("distance"),
        )
        .where(Entity.embedding.is_not(None))
        .order_by("distance")
        .limit(limit)
    )
    out = []
    for row in session.execute(stmt):
        out.append(
            {
                "id": row.id,
                "canonical_name": row.canonical_name,
                "type": row.type,
                "score": round(1.0 - float(row.distance), 4),
            }
        )
    return out
