"""CUSIP extraction with check-digit validation.

A CUSIP is a 9-character security identifier: 8 alphanumeric characters + 1
check digit (computed with a Luhn-style mod-10 over base-36 letter values).
EDGAR text often renders it with internal spaces (e.g. ``35W 10 8``), so we
gather candidate runs near the ``(CUSIP Number)`` label, strip whitespace, and
**validate the check digit** to reject malformed grabs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Candidate: up to ~12 chars of alphanumerics/spaces immediately before the label.
_LABEL_RE = re.compile(r"([0-9A-Za-z][0-9A-Za-z ]{6,14}?)\s*\(\s*CUSIP\s*(?:Number|No\.?)?\s*\)")
_COMPACT_RE = re.compile(r"\b([0-9A-Z]{3}[0-9A-Z]{2}[0-9A-Z]{3}[0-9])\b")


@dataclass(slots=True)
class CusipSpan:
    cusip: str  # normalized 9-char CUSIP
    start: int  # offset of the matched surface in the source text
    end: int
    surface: str  # exact source substring (round-trips)
    valid: bool


def _char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch.upper()) - ord("A") + 10
    # '*'=36, '@'=37, '#'=38 per CUSIP spec; rare.
    return {"*": 36, "@": 37, "#": 38}.get(ch, 0)


def cusip_check_digit(first8: str) -> int:
    total = 0
    for i, ch in enumerate(first8):
        v = _char_value(ch)
        if i % 2 == 1:  # double every second char (0-indexed odd positions)
            v *= 2
        total += v // 10 + v % 10
    return (10 - (total % 10)) % 10


def is_valid_cusip(cusip: str) -> bool:
    c = cusip.strip().upper()
    if len(c) != 9 or not re.fullmatch(r"[0-9A-Z*@#]{9}", c):
        return False
    return cusip_check_digit(c[:8]) == _char_value(c[8])


def extract_cusip(text: str) -> CusipSpan | None:
    """Find the issuer CUSIP near the cover-page label, validating the check digit."""
    best: CusipSpan | None = None
    for m in _LABEL_RE.finditer(text):
        surface = m.group(1)
        compact = re.sub(r"\s+", "", surface).upper()
        if len(compact) != 9:
            continue
        span = CusipSpan(
            cusip=compact,
            start=m.start(1),
            end=m.start(1) + len(surface.rstrip()),
            surface=text[m.start(1) : m.start(1) + len(surface.rstrip())],
            valid=is_valid_cusip(compact),
        )
        # Prefer a valid CUSIP; otherwise keep the first candidate seen.
        if span.valid:
            return span
        best = best or span
    return best
