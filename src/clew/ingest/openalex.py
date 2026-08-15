"""OpenAlex connector (Pack B) — scholarly works + citation graph (no API key).

Each work becomes a ``RawDocument`` whose normalized ``text`` is a deterministic
rendering of title / abstract / authors / affiliations / references / concepts.
All Pack-B claim evidence offsets index into this rendering, preserving the same
provenance invariant used for filings (``text[start:end] == snippet``).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx

from clew.config import get_settings
from clew.ingest.base import Connector, RawDocument

OPENALEX = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _short_id(openalex_url: str | None) -> str | None:
    if not openalex_url:
        return None
    return openalex_url.rstrip("/").split("/")[-1]


def _doi(doi_url: str | None) -> str | None:
    """Normalize a DOI by stripping the resolver prefix (DOIs contain slashes)."""
    if not doi_url:
        return None
    return doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")


def render_work(work: dict) -> tuple[str, dict]:
    """Return (normalized_text, structured_meta) for a work."""
    title = work.get("title") or work.get("display_name") or ""
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    authors = []
    for a in work.get("authorships", []):
        au = a.get("author", {})
        authors.append(
            {
                "id": _short_id(au.get("id")),
                "name": au.get("display_name") or "",
                "orcid": _short_id(au.get("orcid")) if au.get("orcid") else None,
                "institutions": [
                    {"id": _short_id(i.get("id")), "name": i.get("display_name"),
                     "ror": _short_id(i.get("ror")) if i.get("ror") else None}
                    for i in a.get("institutions", [])
                ],
            }
        )
    references = [_short_id(r) for r in work.get("referenced_works", [])]
    concepts = [
        {"id": _short_id(c.get("id")), "name": c.get("display_name")}
        for c in work.get("concepts", [])[:8]
    ]

    # Deterministic text rendering (offsets index into this).
    lines = [f"TITLE: {title}", ""]
    if abstract:
        lines += [f"ABSTRACT: {abstract}", ""]
    lines.append("AUTHORS: " + "; ".join(a["name"] for a in authors if a["name"]))
    aff_pairs = [
        f"{a['name']} — {inst['name']}"
        for a in authors
        for inst in a["institutions"]
        if a["name"] and inst.get("name")
    ]
    if aff_pairs:
        lines.append("AFFILIATIONS: " + "; ".join(aff_pairs))
    if concepts:
        lines.append("CONCEPTS: " + "; ".join(c["name"] for c in concepts if c["name"]))
    if references:
        lines.append("REFERENCES: " + "; ".join(references))
    text = "\n".join(lines)

    meta = {
        "openalex": _short_id(work.get("id")),
        "doi": _doi(work.get("doi")),
        "title": title,
        "publication_date": work.get("publication_date"),
        "authors": authors,
        "references": references,
        "concepts": concepts,
    }
    return text, meta


class OpenAlexConnector(Connector):
    source_name = "OpenAlex"

    def __init__(self, mailto: str | None = None) -> None:
        # Reuse the SEC user-agent email as the OpenAlex polite-pool mailto.
        ua = mailto or get_settings().sec_user_agent
        self.mailto = ua.split()[-1] if "@" in ua else "research@example.com"

    def fetch(
        self,
        limit: int,
        *,
        concept_id: str | None = None,
        search: str | None = None,
        from_date: str | None = None,
    ) -> Iterator[RawDocument]:
        per_page = min(limit, 200)
        params: dict = {"per_page": per_page, "mailto": self.mailto}
        filters = []
        if concept_id:
            filters.append(f"concepts.id:{concept_id}")
        if from_date:
            filters.append(f"from_publication_date:{from_date}")
        if filters:
            params["filter"] = ",".join(filters)
        if search:
            params["search"] = search

        emitted = 0
        cursor = "*"
        while emitted < limit:
            params["cursor"] = cursor
            resp = httpx.get(OPENALEX, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for work in results:
                if emitted >= limit:
                    break
                text, meta = render_work(work)
                if not meta["openalex"]:
                    continue
                emitted += 1
                yield RawDocument(
                    external_id=meta["openalex"],
                    doc_type="openalex_work",
                    url=work.get("id"),
                    retrieved_at=datetime.now(UTC),
                    text=text,
                    meta=meta,
                )
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
