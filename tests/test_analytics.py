"""Unit tests for graph analytics on a toy ownership graph (no DB)."""

from __future__ import annotations

import networkx as nx

from clew.analytics.graph_metrics import centrality, communities, interlocks, summary


def _toy() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n, label, typ in [
        ("F1", "Fund One LP", "Organization"),
        ("F2", "Fund Two LP", "Organization"),
        ("I1", "Issuer A", "Organization"),
        ("I2", "Issuer B", "Organization"),
        ("I3", "Issuer C", "Organization"),
    ]:
        g.add_node(n, label=label, type=typ)
    # F1 owns I1, I2 (interlock); F2 owns I2 (shared issuer with F1).
    g.add_edge("F1", "I1", predicate="OWNS")
    g.add_edge("F1", "I2", predicate="OWNS")
    g.add_edge("F2", "I2", predicate="OWNS")
    return g


def test_interlocks_finds_multi_issuer_holder():
    g = _toy()
    locks = interlocks(g, min_targets=2)
    ids = {x["id"] for x in locks}
    assert "F1" in ids  # owns two issuers
    assert "F2" not in ids  # owns only one


def test_centrality_ranks_shared_issuer_high():
    g = _toy()
    pr = centrality(g, metric="pagerank", limit=5)
    # I2 (owned by both funds) should outrank I1 (owned by one).
    order = [r["id"] for r in pr]
    assert order.index("I2") < order.index("I1")


def test_communities_and_summary():
    g = _toy()
    comms = communities(g)
    assert sum(c["size"] for c in comms) == g.number_of_nodes()
    s = summary(g)
    assert s["nodes"] == 5 and s["edges"] == 3 and s["interlocks"] == 1
