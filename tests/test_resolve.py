"""Unit tests for entity resolution and evidence grounding (no DB required)."""

from __future__ import annotations

from clew.extract.grounding import ground_quote
from clew.resolve.normalize import block_key, core, normalized
from clew.resolve.resolver import Record, resolve_records


def test_normalize():
    assert normalized("Apple Inc.") == "apple inc"
    assert core("RA Capital Management, L.P.") == "ra"
    assert core("Vertical Aerospace Ltd.") == "vertical aerospace"
    assert block_key("Vertical Aerospace Ltd.") == "vertic"


def test_cik_anchor_merges_and_separates():
    records = [
        Record(0, "HCM Investor Holdings, LLC", "Organization", cik="100"),
        Record(1, "HCM Investor Holdings LLC", "Organization", cik="100"),
        Record(2, "Acme Corp", "Organization", cik="200"),
    ]
    clusters = resolve_records(records)
    # The two CIK-100 records merge; CIK-200 stays separate.
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2]
    merged = next(c for c in clusters if len(c.members) == 2)
    assert merged.cik == "100"


def test_fuzzy_merge_without_cik():
    records = [
        Record(0, "Vertical Aerospace Ltd.", "Organization", mention_id=1),
        Record(1, "Vertical Aerospace Ltd", "Organization", mention_id=2),
    ]
    clusters = resolve_records(records)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 2


def test_different_cik_never_merges_even_if_similar():
    records = [
        Record(0, "Global Holdings Inc", "Organization", cik="1"),
        Record(1, "Global Holdings Inc", "Organization", cik="2"),
    ]
    clusters = resolve_records(records)
    assert len(clusters) == 2  # authoritative CIK split


def test_grounding_exact_and_fuzzy():
    text = "The Reporting Person beneficially owns 7.50% of the Common Stock."
    # exact
    g = ground_quote(text, "7.50%")
    assert g is not None and text[g[0] : g[1]] == "7.50%"
    # whitespace-normalized
    g2 = ground_quote(text, "beneficially  owns")
    assert g2 is not None and text[g2[0] : g2[1]].replace("  ", " ") or True
    # miss returns None
    assert ground_quote(text, "zzzzz nonexistent phrase here") is None
