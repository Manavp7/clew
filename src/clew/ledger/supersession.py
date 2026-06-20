"""Supersession reconciliation — turn a stream of filings into clean valid-time.

A 13D/A amendment restates an ownership stake at a later date. Rather than match
accession chains, we infer supersession directly from the ledger: for each
``(subject, predicate, object)`` group, order the claims by valid time and link
each claim to the next one that starts later, closing the earlier claim's
``valid_to`` at the successor's ``valid_from``.

Effect:
* **Clean current snapshot** — valid-as-of(today) returns only the latest stake.
* **Full history preserved** — superseded claims keep their (now closed) valid
  interval and a ``superseded_by`` pointer; nothing is deleted (append-only).
* **Contradictions de-noised** — adjacent (touching) intervals no longer overlap,
  so stake *changes* stop being flagged as conflicts; only same-period conflicts
  remain.

This mutates only the designed transition columns (``valid_to``,
``superseded_by``), consistent with the bitemporal schema.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from clew.db.models import Claim


def reconcile_supersessions(session: Session) -> dict:
    """Link consecutive same-(subject,predicate,object) claims by valid time."""
    claims = session.execute(
        select(Claim).where(Claim.retracted_at.is_(None))
    ).scalars().all()

    groups: dict[tuple, list[Claim]] = defaultdict(list)
    for c in claims:
        if c.valid_from is None:
            continue  # cannot order without a world-time anchor
        groups[(c.subject_id, c.predicate, c.object_id)].append(c)

    superseded = 0
    chains = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda c: (c.valid_from, c.asserted_at, c.id))
        linked_in_group = False
        for earlier, later in zip(group, group[1:], strict=False):
            if later.valid_from <= earlier.valid_from:
                continue  # same/earlier date -> a contradiction candidate, not supersession
            earlier.superseded_by = later.id
            if earlier.valid_to is None or earlier.valid_to > later.valid_from:
                earlier.valid_to = later.valid_from
            superseded += 1
            linked_in_group = True
        if linked_in_group:
            chains += 1

    session.flush()
    return {"chains": chains, "claims_superseded": superseded}
