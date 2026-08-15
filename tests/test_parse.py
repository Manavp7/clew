"""Unit tests for offset-grounded 13D/13G cover parsing (no DB required)."""

from __future__ import annotations

from clew.parse.cusip import cusip_check_digit, extract_cusip, is_valid_cusip
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


def test_cusip_check_digit_and_validation():
    # Real, well-known CUSIPs.
    assert cusip_check_digit("03783310") == 0  # Apple 037833100
    assert is_valid_cusip("037833100")
    assert is_valid_cusip("38259P508")  # Alphabet/Google
    assert not is_valid_cusip("12345A678")  # wrong check digit
    assert not is_valid_cusip("123")  # too short


def test_extract_cusip_normalizes_internal_spaces():
    text = "(Title of Class of Securities)\n\n82835W 10 8\n\n(CUSIP Number)\n"
    cs = extract_cusip(text)
    assert cs is not None
    assert cs.cusip == "82835W108" and cs.valid
    # the source surface still round-trips to its offsets
    assert text[cs.start : cs.end] == cs.surface


def test_parse_filing_uses_validated_cusip():
    parsed = parse_filing(SAMPLE)
    assert parsed.cusip_value == "12345A678"
    # SAMPLE's CUSIP has an invalid check digit -> flagged but still captured.
    assert parsed.cusip_valid is False
