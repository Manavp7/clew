"""Entity merge operation — re-point all references from one entity to another.

Merging two canonical entities is a correction to the *canonical layer*, not a
mutation of claims: every claim/mention is re-pointed to the kept entity, the
kept entity absorbs the dropped entity's aliases + external ids, the merge is
logged, and the dropped (duplicate) entity row is removed. Claims (the
statements) are untouched, so the ledger's append-only guarantee holds.
"""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from clew.db.models import Claim, Entity, EntityMergeLog, Mention


class MergeError(ValueError):
    pass


def merge_entities(session: Session, keep_id: str, drop_id: str) -> dict:
    """Merge ``drop_id`` into ``keep_id``. Returns a summary of what was re-pointed."""
    if keep_id == drop_id:
        raise MergeError("cannot merge an entity into itself")
    keep = session.get(Entity, keep_id)
    drop = session.get(Entity, drop_id)
    if keep is None or drop is None:
        raise MergeError(f"entity not found: {keep_id if keep is None else drop_id}")
    if keep.type != drop.type:
        raise MergeError(f"type mismatch: {keep.type} != {drop.type}")

    # Re-point claims (subject + object) and mentions.
    claims_subj = session.execute(
        update(Claim).where(Claim.subject_id == drop_id).values(subject_id=keep_id)
    ).rowcount
    claims_obj = session.execute(
        update(Claim).where(Claim.object_id == drop_id).values(object_id=keep_id)
    ).rowcount
    mentions = session.execute(
        update(Mention).where(Mention.resolved_to == drop_id).values(resolved_to=keep_id)
    ).rowcount

    # Absorb aliases + external ids.
    merged_aliases = set(keep.aliases or []) | set(drop.aliases or []) | {drop.canonical_name}
    merged_aliases.discard(keep.canonical_name)
    keep.aliases = sorted(merged_aliases)
    keep.external_ids = {**(drop.external_ids or {}), **(keep.external_ids or {})}

    session.add(
        EntityMergeLog(
            kept_id=keep_id,
            dropped_id=drop_id,
            dropped_name=drop.canonical_name,
            claims_repointed=claims_subj + claims_obj,
            mentions_repointed=mentions,
        )
    )
    session.delete(drop)
    session.flush()
    return {
        "kept": keep_id,
        "dropped": drop_id,
        "claims_repointed": claims_subj + claims_obj,
        "mentions_repointed": mentions,
    }


def _entity_claim_count(session: Session, entity_id: str) -> int:
    n_subj = len(
        session.execute(select(Claim.id).where(Claim.subject_id == entity_id)).scalars().all()
    )
    n_obj = len(
        session.execute(select(Claim.id).where(Claim.object_id == entity_id)).scalars().all()
    )
    return n_subj + n_obj
