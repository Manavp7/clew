"""Entity resolution: cluster surface records into canonical entities.

Strategy (high-precision first):

1. **CIK anchor (deterministic).** Records sharing a CIK are the same entity;
   records with *different* non-null CIKs are never merged. This is the backbone
   that prevents ER from poisoning the graph.
2. **Fuzzy linkage (Fellegi-Sunter-style).** Within a cheap blocking key, link
   records whose normalized names exceed a Jaro-Winkler threshold. ``resolution_
   confidence`` carries the similarity score; CIK-anchored links score 1.0.

The clustering uses union-find. Splink can be slotted in as the probabilistic
linkage backend (see ``splink_er.py``) once data volume supports EM training;
this resolver is the robust Phase-1 default and shares the same output contract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import distance

from clew.resolve.normalize import block_key, core, normalized

JW_THRESHOLD = 0.92


@dataclass(slots=True)
class Record:
    rid: int  # local record id
    name: str
    entity_type: str
    cik: str | None = None
    mention_id: int | None = None  # None for header-derived records


@dataclass(slots=True)
class Cluster:
    canonical_name: str
    entity_type: str
    cik: str | None
    aliases: list[str]
    members: list[Record] = field(default_factory=list)
    confidence: float = 1.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _jw(a: str, b: str) -> float:
    return 1.0 - distance.JaroWinkler.normalized_distance(a, b)


def resolve_records(
    records: list[Record],
    *,
    must_link: set[frozenset[str]] | None = None,
    must_not_link: set[frozenset[str]] | None = None,
) -> list[Cluster]:
    """Cluster records into canonical entities. Pure function (no DB).

    ``must_link`` / ``must_not_link`` are sets of ``frozenset({core_a, core_b})``
    from human ER feedback: must-link pairs are always unioned, must-not-link
    pairs are never unioned (human corrections override string similarity).
    """
    must_link = must_link or set()
    must_not_link = must_not_link or set()
    n = len(records)
    uf = _UnionFind(n)
    norms = [normalized(r.name) for r in records]
    cores = [core(r.name) for r in records]

    def _blocked(i: int, j: int) -> bool:
        return frozenset({cores[i], cores[j]}) in must_not_link

    # 1) CIK anchor: union all records sharing a CIK.
    by_cik: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        if r.cik:
            by_cik.setdefault(r.cik, []).append(i)
    for idxs in by_cik.values():
        for j in idxs[1:]:
            if not _blocked(idxs[0], j):
                uf.union(idxs[0], j)

    # 2) Fuzzy linkage within blocks; never cross differing CIKs, types, or must-not-link.
    blocks: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(records):
        blocks.setdefault((r.entity_type, block_key(r.name)), []).append(i)
    for idxs in blocks.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                ci, cj = records[i].cik, records[j].cik
                if ci and cj and ci != cj:
                    continue  # authoritative: different CIK => different entity
                if _blocked(i, j):
                    continue  # human must-not-link
                if _jw(norms[i], norms[j]) >= JW_THRESHOLD:
                    uf.union(i, j)

    # 3) Human must-link: union any pair whose core names match a must-link decision.
    if must_link:
        for a in range(n):
            for b in range(a + 1, n):
                if records[a].entity_type != records[b].entity_type:
                    continue
                if frozenset({cores[a], cores[b]}) in must_link and not _blocked(a, b):
                    uf.union(a, b)

    # Gather clusters.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    clusters: list[Cluster] = []
    for idxs in groups.values():
        members = [records[i] for i in idxs]
        names = [m.name for m in members]
        canonical_name = _pick_canonical(names)
        ciks = {m.cik for m in members if m.cik}
        cik = next(iter(ciks)) if ciks else None
        aliases = sorted({n for n in names if n != canonical_name})
        # Confidence: 1.0 if CIK-anchored or single record; else min pairwise JW proxy.
        conf = 1.0 if (cik or len(members) == 1) else _cluster_confidence(members, norms, idxs)
        clusters.append(
            Cluster(
                canonical_name=canonical_name,
                entity_type=members[0].entity_type,
                cik=cik,
                aliases=aliases,
                members=members,
                confidence=conf,
            )
        )
    return clusters


def _pick_canonical(names: list[str]) -> str:
    """Most frequent surface; ties broken by longest (most descriptive)."""
    counts = Counter(names)
    best = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))
    return best[0]


def _cluster_confidence(members, norms, idxs) -> float:
    if len(idxs) < 2:
        return 1.0
    scores = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            scores.append(_jw(norms[idxs[a]], norms[idxs[b]]))
    return round(min(scores), 3) if scores else 1.0


def normalize_for_index(name: str) -> str:
    """Key used to look up an entity by surface name (claim subject/object)."""
    return core(name)
