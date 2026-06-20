"""Pack A — Investigative / Financial domain pack.

Defines the controlled vocabulary (entity types, predicates, qualifiers) for the
13D/13G beneficial-ownership wedge. Predicates are aligned with the OCCRP
*Follow-the-Money* (FtM) ontology so the model maps cleanly onto established
investigative tooling later.

This registry is the contract the extractor must produce against, and what the
projection/eval layers validate.
"""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    ORGANIZATION = "Organization"
    PERSON = "Person"
    SECURITY = "Security"  # the class of stock being reported on


class Predicate(StrEnum):
    """Relationship predicates for the financial pack (FtM-aligned)."""

    # Beneficial ownership / control (the 13D/13G core)
    OWNS = "OWNS"  # subject holds an equity stake in object (qualifier: stake_pct)
    CONTROLS = "CONTROLS"  # voting/dispositive control
    BENEFICIAL_OWNER_OF = "BENEFICIAL_OWNER_OF"
    # Corporate structure / people
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PARENT_OF = "PARENT_OF"
    BOARD_MEMBER_OF = "BOARD_MEMBER_OF"
    OFFICER_OF = "OFFICER_OF"
    AFFILIATE_OF = "AFFILIATE_OF"
    # Events
    ACQUIRED = "ACQUIRED"
    FILED = "FILED"  # subject (filer) FILED object (the filing/issuer context)


# Qualifier keys recognised on claims, with light typing hints used by eval.
QUALIFIER_SCHEMA: dict[str, str] = {
    "stake_pct": "number",  # percent of class beneficially owned
    "shares": "number",  # number of shares
    "voting_power": "number",  # shares with sole/shared voting power
    "dispositive_power": "number",
    "price": "number",
    "currency": "string",
    "security_class": "string",  # e.g. "Common Stock"
    "cusip": "string",
    "event_date": "date",  # date of the triggering transaction
}


# Predicates whose object is a literal value rather than an entity.
LITERAL_OBJECT_PREDICATES: frozenset[Predicate] = frozenset()

# GLiNER zero-shot labels used for mention detection in this pack.
NER_LABELS: list[str] = [
    "organization",
    "person",
    "percentage",
    "monetary amount",
    "date",
    "security class",
]

# Map external-id namespaces to the entity types they anchor (for ER).
EXTERNAL_ID_ANCHORS: dict[str, EntityType] = {
    "cik": EntityType.ORGANIZATION,  # SEC Central Index Key
    "lei": EntityType.ORGANIZATION,  # GLEIF Legal Entity Identifier
    "cusip": EntityType.SECURITY,
    "irs_no": EntityType.ORGANIZATION,
}

PACK_NAME = "pack_a_financial"


def is_valid_predicate(value: str) -> bool:
    return value in Predicate._value2member_map_


def is_valid_entity_type(value: str) -> bool:
    return value in EntityType._value2member_map_
