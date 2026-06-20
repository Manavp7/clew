"""Pack B — Scientific Discovery domain pack.

Demonstrates the **domain-agnostic core**: the ledger, entity resolution,
projections, analytics, and eval are reused unchanged; only this pack
(ontology + connector + extractor) is added.

Claims are nanopublication-style — a scientific statement (AUTHORED, CITES,
AFFILIATED_WITH, ...) + provenance (evidence span) + attribution (extractor).
External-id anchors (OpenAlex ID, DOI, ORCID, ROR) give deterministic ER, the
same machinery CIK provides in Pack A.
"""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"
    CONCEPT = "Concept"


class Predicate(StrEnum):
    AUTHORED = "AUTHORED"  # author -> paper
    CITES = "CITES"  # paper -> paper
    AFFILIATED_WITH = "AFFILIATED_WITH"  # author -> institution
    MENTIONS_CONCEPT = "MENTIONS_CONCEPT"  # paper -> concept


# External-id namespaces -> the entity type they anchor (deterministic ER).
EXTERNAL_ID_ANCHORS: dict[str, EntityType] = {
    "openalex": EntityType.PAPER,
    "doi": EntityType.PAPER,
    "openalex_author": EntityType.AUTHOR,
    "orcid": EntityType.AUTHOR,
    "openalex_institution": EntityType.INSTITUTION,
    "ror": EntityType.INSTITUTION,
    "openalex_concept": EntityType.CONCEPT,
}

PACK_NAME = "pack_b_science"


def is_valid_predicate(value: str) -> bool:
    return value in Predicate._value2member_map_
