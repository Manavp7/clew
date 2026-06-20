"""Graph projection — NetworkX, rebuilt from the ledger (no Neo4j in Phase 1).

The graph is a *projection*: nodes are entities referenced by claims, edges are
claims (believed as of a transaction time). It can be rebuilt from scratch at any
time and is exported as Sigma.js-friendly JSON.
"""

from __future__ import annotations

from datetime import date, datetime

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Entity
from clew.ledger.asof import claims_asof


def build_graph(
    session: Session, *, as_of: datetime | None = None, valid_on: date | None = "today"
) -> nx.MultiDiGraph:
    """Build a directed multigraph of entities + claim edges.

    By default the graph reflects the **current world state** (valid as of today),
    so superseded ownership stakes drop out and only the latest stake per
    relationship is shown. Pass ``valid_on=None`` to include all valid intervals.
    """
    graph = nx.MultiDiGraph()
    if valid_on == "today":
        valid_on = date.today()
    claims = claims_asof(session, as_of=as_of, valid_on=valid_on)

    needed: set[str] = set()
    for c in claims:
        needed.add(c.subject_id)
        if c.object_id:
            needed.add(c.object_id)

    if needed:
        entities = session.execute(
            select(Entity).where(Entity.id.in_(needed))
        ).scalars().all()
        for e in entities:
            graph.add_node(
                e.id,
                label=e.canonical_name,
                type=e.type,
                cik=(e.external_ids or {}).get("cik"),
                aliases=e.aliases or [],
            )

    for c in claims:
        if not c.object_id:
            continue  # literal-object claims are node attributes, not edges (Phase 1)
        graph.add_edge(
            c.subject_id,
            c.object_id,
            key=c.id,
            claim_id=c.id,
            predicate=c.predicate,
            stake_pct=(c.qualifiers or {}).get("stake_pct"),
            valid_from=c.valid_from.isoformat() if c.valid_from else None,
            valid_to=c.valid_to.isoformat() if c.valid_to else None,
            confidence=c.confidence,
            evidence_count=len(c.evidence),
        )
    return graph


def graph_to_json(graph: nx.MultiDiGraph) -> dict:
    """Sigma.js-friendly node/edge JSON."""
    nodes = [
        {
            "id": n,
            "label": d.get("label", n),
            "type": d.get("type"),
            "cik": d.get("cik"),
            "degree": graph.degree(n),
        }
        for n, d in graph.nodes(data=True)
    ]
    edges = [
        {
            "id": f"{u}-{v}-{k}",
            "source": u,
            "target": v,
            "predicate": d.get("predicate"),
            "stake_pct": d.get("stake_pct"),
            "valid_from": d.get("valid_from"),
            "claim_id": d.get("claim_id"),
            "evidence_count": d.get("evidence_count"),
        }
        for u, v, k, d in graph.edges(keys=True, data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def ego_graph(graph: nx.MultiDiGraph, center: str, radius: int = 1) -> nx.MultiDiGraph:
    """Neighborhood subgraph around ``center`` (undirected reachability)."""
    if center not in graph:
        return nx.MultiDiGraph()
    und = graph.to_undirected(as_view=True)
    nodes = nx.ego_graph(und, center, radius=radius).nodes()
    return graph.subgraph(nodes).copy()
