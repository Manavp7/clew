"""Graph analytics over the rebuildable NetworkX projection (no Neo4j).

All metrics are computed on the *current-snapshot* ownership graph (a projection
of the ledger), so they're reproducible and never a separate source of truth:

* **centrality** — degree / PageRank / betweenness (who is most connected/influential),
* **communities** — greedy-modularity clusters on the undirected projection,
* **interlocks** — filers connected to ≥2 issuers (and, once BOARD_MEMBER_OF
  claims exist, people who sit on/control multiple organizations).
"""

from __future__ import annotations

import networkx as nx


def _node_label(g: nx.MultiDiGraph, n: str) -> str:
    return g.nodes[n].get("label", n)


def centrality(g: nx.MultiDiGraph, *, metric: str = "pagerank", limit: int = 20) -> list[dict]:
    if g.number_of_nodes() == 0:
        return []
    if metric == "degree":
        scores = dict(g.degree())
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(g)
    else:  # pagerank (default)
        # PageRank on a simple directed view (collapse multi-edges).
        simple = nx.DiGraph()
        simple.add_nodes_from(g.nodes())
        for u, v in g.edges():
            simple.add_edge(u, v)
        scores = nx.pagerank(simple) if simple.number_of_edges() else dict(g.degree())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [
        {
            "id": n,
            "label": _node_label(g, n),
            "type": g.nodes[n].get("type"),
            "score": round(float(s), 6),
        }
        for n, s in ranked
    ]


def communities(g: nx.MultiDiGraph, *, limit: int = 25) -> list[dict]:
    if g.number_of_nodes() == 0:
        return []
    und = nx.Graph()
    und.add_nodes_from(g.nodes(data=True))
    for u, v in g.edges():
        und.add_edge(u, v)
    comms = nx.community.greedy_modularity_communities(und) if und.number_of_edges() else []
    out: list[dict] = []
    for i, members in enumerate(comms[:limit]):
        members = list(members)
        out.append(
            {
                "community": i,
                "size": len(members),
                "members": [
                    {"id": n, "label": _node_label(g, n), "type": g.nodes[n].get("type")}
                    for n in members
                ],
            }
        )
    return out


def interlocks(g: nx.MultiDiGraph, *, min_targets: int = 2, limit: int = 50) -> list[dict]:
    """Filers/holders connected to multiple distinct issuers (ownership interlocks)."""
    out: list[dict] = []
    for n in g.nodes():
        targets = {v for _, v in g.out_edges(n)}
        if len(targets) >= min_targets:
            out.append(
                {
                    "id": n,
                    "label": _node_label(g, n),
                    "type": g.nodes[n].get("type"),
                    "issuer_count": len(targets),
                    "issuers": [
                        {"id": t, "label": _node_label(g, t)} for t in list(targets)[:25]
                    ],
                }
            )
    out.sort(key=lambda d: d["issuer_count"], reverse=True)
    return out[:limit]


def summary(g: nx.MultiDiGraph) -> dict:
    und = nx.Graph()
    und.add_nodes_from(g.nodes())
    for u, v in g.edges():
        und.add_edge(u, v)
    n_comms = (
        len(nx.community.greedy_modularity_communities(und)) if und.number_of_edges() else 0
    )
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "components": nx.number_connected_components(und) if und.number_of_nodes() else 0,
        "communities": n_comms,
        "interlocks": len(interlocks(g)),
    }
