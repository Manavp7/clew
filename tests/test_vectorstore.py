"""Test the embedded Qdrant VectorStore backend (no server, no DB)."""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("qdrant_client")

from clew.project.vectorstore import QdrantStore, VectorHit  # noqa: E402


def test_qdrant_embedded_upsert_and_search():
    store = QdrantStore(path=tempfile.mkdtemp(), dim=4)
    store.upsert(
        [
            ("ORG_1", [0.1, 0.2, 0.3, 0.4], {"canonical_name": "Alpha", "type": "Organization"}),
            ("ORG_2", [0.9, 0.1, 0.0, 0.0], {"canonical_name": "Beta", "type": "Organization"}),
        ]
    )
    hits = store.search([0.1, 0.2, 0.3, 0.4], limit=2)
    assert isinstance(hits[0], VectorHit)
    # Nearest neighbour is the matching vector, returned by its entity id.
    assert hits[0].id == "ORG_1"
    assert hits[0].payload["canonical_name"] == "Alpha"
    assert hits[0].score >= hits[1].score
