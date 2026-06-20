"""Claim extractors producing the provider-agnostic ``ExtractedClaim`` IR.

* :class:`RuleClaimExtractor` — deterministic baseline. Builds
  ``OWNS(filer -> issuer, stake_pct)`` from the structured filing header (CIK
  anchors) + the offset-grounded cover percentage. Always available; used for
  eval comparison and degraded (no-key) operation.
* :class:`LLMClaimExtractor` — primary path. Uses the provider abstraction
  (Vercel AI Gateway) to extract a richer claim set with verbatim evidence
  quotes, grounded to offsets afterwards.
"""

from __future__ import annotations

from datetime import datetime

from clew.extract.schemas import ExtractedClaim, ExtractedClaims
from clew.llm.base import LLMClient
from clew.packs.pack_a_financial import Predicate
from clew.parse.edgar import ParsedFiling


def _event_date_iso(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


class RuleClaimExtractor:
    method = "rule"
    version = "rule@0.1"

    def extract(self, *, text: str, parsed: ParsedFiling, meta: dict) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        filers = [f for f in meta.get("filers", []) if f.get("name")]
        subjects = [s for s in meta.get("subject_companies", []) if s.get("name")]
        bp = parsed.best_percent
        valid_from = _event_date_iso(parsed.event_date_value)

        base_quals: dict = {}
        if parsed.event_date_value:
            base_quals["event_date"] = parsed.event_date_value
        if parsed.cusip:
            base_quals["cusip"] = parsed.cusip.surface

        for subj in subjects:
            for filer in filers:
                quals = dict(base_quals)
                if bp is not None:
                    quals["stake_pct"] = bp.value
                    evidence_quote = bp.span.surface
                elif parsed.issuer_name is not None:
                    evidence_quote = parsed.issuer_name.surface
                else:
                    continue
                claims.append(
                    ExtractedClaim(
                        subject_surface=filer["name"],
                        predicate=Predicate.OWNS.value,
                        object_surface=subj["name"],
                        qualifiers=quals,
                        valid_from=valid_from,
                        confidence=0.9 if bp is not None else 0.6,
                        evidence_quote=evidence_quote,
                    )
                )
        return claims


_SYSTEM = (
    "You are an expert financial-filings analyst extracting beneficial-ownership "
    "facts from SEC Schedule 13D/13G filings. Extract only claims explicitly "
    "supported by the text. For each claim provide a VERBATIM evidence_quote "
    "copied exactly from the document. Use only these predicates: "
    f"{[p.value for p in Predicate]}. Express ownership stake as qualifiers."
    "stake_pct (a number, percent of class)."
)


class LLMClaimExtractor:
    method = "llm-extract"

    def __init__(self, client: LLMClient | None = None) -> None:
        if client is None:
            from clew.llm.gateway import get_extractor

            client = get_extractor()
        self.client = client
        self.version = f"llm-extract@{client.model}"

    def extract(self, *, text: str, parsed: ParsedFiling, meta: dict) -> list[ExtractedClaim]:
        filer_names = [f.get("name") for f in meta.get("filers", []) if f.get("name")]
        subject_names = [s.get("name") for s in meta.get("subject_companies", []) if s.get("name")]
        hint = (
            f"Reporting persons (filers): {filer_names}. "
            f"Subject company (issuer): {subject_names}.\n\n"
        )
        # Cover page + Item 5 carry the ownership facts; cap to a generous window.
        body = text[:18000]
        user = hint + "Filing text:\n" + body
        result: ExtractedClaims = self.client.extract(_SYSTEM, user, ExtractedClaims)
        return result.claims
