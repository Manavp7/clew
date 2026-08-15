"""Pluggable vector store behind one interface (a rebuildable projection).

* :class:`PgVectorStore` (default) — vectors live on ``entity.embedding`` in
  Postgres; search via pgvector cosine distance.
* :class:`QdrantStore` — embedded `qdrant-client` (on-disk, **no server**, so it
  runs without Docker); search via Qdrant cosine similarity.

Both are *projections* of the ledger and fully rebuildable. Backend is selected
by ``CLEW_VECTOR_BACKEND``; swapping is config-only — no calling code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from clew.config import get_settings


@dataclass(slots=True)
class VectorHit:
    id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    def upsert(self, items: list[tuple[str, list[float], dict]]) -> int: ...
    def search(self, vector: list[float], *, limit: int = 10) -> list[VectorHit]: ...


class PgVectorStore:
    """Stores vectors on entity rows; searches with pgvector cosine distance."""

    COLLECTION = "entity"

    def upsert(self, items: list[tuple[str, list[float], dict]]) -> int:
        from clew.db.models import Entity
        from clew.db.session import write_session

        n = 0
        with write_session() as session:
            for eid, vec, _payload in items:
                e = session.get(Entity, eid)
                if e is not None:
                    e.embedding = vec
                    n += 1
        return n

    def search(self, vector: list[float], *, limit: int = 10) -> list[VectorHit]:
        from sqlalchemy import select

        from clew.db.models import Entity
        from clew.db.session import read_session

        with read_session() as session:
            stmt = (
                select(
                    Entity.id,
                    Entity.canonical_name,
                    Entity.type,
                    Entity.embedding.cosine_distance(vector).label("distance"),
                )
                .where(Entity.embedding.is_not(None))
                .order_by("distance")
                .limit(limit)
            )
            return [
                VectorHit(
                    id=row.id,
                    score=round(1.0 - float(row.distance), 4),
                    payload={"canonical_name": row.canonical_name, "type": row.type},
                )
                for row in session.execute(stmt)
            ]


class QdrantStore:
    """Embedded Qdrant (on-disk, no server)."""

    COLLECTION = "entities"

    def __init__(self, path: str | None = None, dim: int | None = None) -> None:
        s = get_settings()
        self.path = path or s.qdrant_path
        self.dim = dim or s.embedding_dim
        self._client = None

    def _c(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(path=self.path)
            if not self._client.collection_exists(self.COLLECTION):
                self._client.create_collection(
                    self.COLLECTION,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
        return self._client

    def upsert(self, items: list[tuple[str, list[float], dict]]) -> int:
        from qdrant_client.models import PointStruct

        client = self._c()
        points = [
            PointStruct(id=_stable_int_id(eid), vector=vec, payload={"entity_id": eid, **payload})
            for eid, vec, payload in items
        ]
        if points:
            client.upsert(self.COLLECTION, points=points)
        return len(points)

    def search(self, vector: list[float], *, limit: int = 10) -> list[VectorHit]:
        client = self._c()
        res = client.query_points(self.COLLECTION, query=vector, limit=limit).points
        return [
            VectorHit(
                id=p.payload.get("entity_id", str(p.id)),
                score=round(float(p.score), 4),
                payload={k: v for k, v in p.payload.items() if k != "entity_id"},
            )
            for p in res
        ]


def _stable_int_id(entity_id: str) -> int:
    # Qdrant point ids must be int/uuid; derive a stable int from the entity id.
    import hashlib

    return int(hashlib.sha1(entity_id.encode()).hexdigest()[:15], 16)


def get_vector_store() -> VectorStore:
    s = get_settings()
    if s.vector_backend == "qdrant":
        return QdrantStore()
    return PgVectorStore()
