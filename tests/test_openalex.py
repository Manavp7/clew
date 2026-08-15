"""Unit tests for OpenAlex rendering + Pack-B claim grounding (no network)."""

from __future__ import annotations

from clew.extract.grounding import ground_quote
from clew.ingest.openalex import _reconstruct_abstract, render_work

WORK = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.1/abc",
    "title": "A Study of Things",
    "publication_date": "2021-05-01",
    "abstract_inverted_index": {"We": [0], "study": [1], "things": [2], "carefully": [3]},
    "authorships": [
        {
            "author": {
                "id": "https://openalex.org/A1",
                "display_name": "Ada Lovelace",
                "orcid": "https://orcid.org/0000-0002-1825-0097",
            },
            "institutions": [
                {"id": "https://openalex.org/I1", "display_name": "Analytical Engine Inst"}
            ],
        }
    ],
    "referenced_works": ["https://openalex.org/W999"],
    "concepts": [{"id": "https://openalex.org/C1", "display_name": "Computing"}],
}


def test_reconstruct_abstract():
    assert _reconstruct_abstract(WORK["abstract_inverted_index"]) == "We study things carefully"


def test_render_work_offsets_and_meta():
    text, meta = render_work(WORK)
    assert meta["openalex"] == "W123"
    assert meta["doi"] == "10.1/abc"
    assert meta["authors"][0]["orcid"] == "0000-0002-1825-0097"
    assert meta["references"] == ["W999"]
    # Every span the extractor relies on grounds to the rendered text.
    for quote in ["Ada Lovelace", "Ada Lovelace — Analytical Engine Inst", "W999", "Computing"]:
        g = ground_quote(text, quote)
        assert g is not None and text[g[0] : g[1]] == g[2]
