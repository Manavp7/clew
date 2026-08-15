"""Vector projection — entity embeddings via the configured VectorStore.

Embeds each canonical entity's name + aliases with the configured embedding
model (Qwen3-Embedding on CPU by default) and upserts the vector into the
selected backend (pgvector or embedded Qdrant). Like the graph, this is a
projection of the ledger and can be rebuilt at will.
"""

from __future__ import annotations

from sqlalchemy import select

from clew.config import get_settings
from clew.db.models import Entity
from clew.db.session import write_session


def _entity_text(e: Entity) -> str:
    parts = [e.canonical_name, *(e.aliases or [])]
    return " | ".join(dict.fromkeys(p for p in parts if p))


def build_entity_vectors(batch_size: int = 64, only_missing: bool = False) -> dict:
    from clew.llm.gateway import get_embedder
    from clew.project.vectorstore import get_vector_store

    embedder = get_embedder()
    store = get_vector_store()
    embedded = 0
    with write_session() as session:
        stmt = select(Entity)
        if only_missing:
            stmt = stmt.where(Entity.embedding.is_(None))
        entities = session.execute(stmt).scalars().all()
        rows = [
            (e.id, _entity_text(e), {"canonical_name": e.canonical_name, "type": e.type})
            for e in entities
        ]
    # Embed + upsert in batches via the configured backend (pgvector | qdrant).
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        vectors = embedder.embed([t for _, t, _ in chunk])
        items = [(eid, vec, payload) for (eid, _, payload), vec in zip(chunk, vectors, strict=True)]
        embedded += store.upsert(items)
    return {
        "entities_embedded": embedded,
        "model": embedder.model,
        "dim": embedder.dim,
        "backend": get_settings().vector_backend,
    }


def search_entities(query: str, *, limit: int = 10) -> list[dict]:
    """Semantic nearest-neighbour search via the configured vector backend."""
    from clew.llm.gateway import get_embedder
    from clew.project.vectorstore import get_vector_store

    embedder = get_embedder()
    store = get_vector_store()
    qvec = embedder.embed([query])[0]
    hits = store.search(qvec, limit=limit)
    return [
        {
            "id": h.id,
            "canonical_name": h.payload.get("canonical_name"),
            "type": h.payload.get("type"),
            "score": h.score,
        }
        for h in hits
    ]
