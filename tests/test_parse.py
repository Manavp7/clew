"""Unit tests for offset-grounded 13D/13G cover parsing (no DB required)."""

from __future__ import annotations

from clew.parse.edgar import parse_filing

SAMPLE = """SCHEDULE 13D

Acme Holdings, Inc.

(Name of Issuer)

Common Stock

(Title of Class of Securities)

12345A678

(CUSIP Number)

March 15, 2024

(Date of Event Which Requires Filing of this Statement)

The Reporting Person beneficially owns 7.50% of the outstanding Common Stock.
"""


def test_parse_cover_fields_and_offsets():
    parsed = parse_filing(SAMPLE)
    assert parsed.issuer_name is not None
    assert parsed.issuer_name.surface == "Acme Holdings, Inc."
    # offset round-trip
    s = parsed.issuer_name
    assert SAMPLE[s.start : s.end] == s.surface

    assert parsed.cusip is not None
    assert parsed.cusip.surface == "12345A678"
    assert SAMPLE[parsed.cusip.start : parsed.cusip.end] == parsed.cusip.surface

    assert parsed.event_date_value == "March 15, 2024"

    bp = parsed.best_percent
    assert bp is not None
    assert bp.value == 7.5
    assert SAMPLE[bp.span.start : bp.span.end] == bp.span.surface


def test_parse_rejects_non_ownership_percent():
    text = "The Note bears interest at a rate of prime plus 3% and is due 2026."
    parsed = parse_filing(text)
    # The only percent is an interest rate -> not a positive ownership candidate.
    assert parsed.best_percent is None
