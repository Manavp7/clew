"""Offset-grounded parsing of 13D/13G filings.

Works on ``document.text`` (the normalized string produced at ingest) and never
mutates it, so every returned offset round-trips: ``text[start:end] == surface``.

Extracts the semi-structured cover-page facts that anchor ownership claims:

* issuer name + CUSIP (the security/issuer being reported on),
* the *event date* (valid-time anchor for the ownership stake),
* candidate beneficial-ownership percentages, **context-scored** to reject
  non-ownership numbers (interest rates, thresholds like "130% of").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clew.parse.cusip import extract_cusip

# Percent like "19.17%", "11.0%", "9.99 %".
_PCT_RE = re.compile(r"\d{1,3}(?:\.\d{1,3})?\s*%")
_DATE_RE = re.compile(
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4})"
)

# Positive context: words that indicate a beneficial-ownership percentage.
_POS = re.compile(
    r"(represent|beneficial|percent of class|of the (?:outstanding|issued|issuer)|"
    r"shares outstanding|aggregate amount|row \(11\))",
    re.I,
)
# Negative context: numbers that look like percentages but are not ownership.
_NEG = re.compile(
    r"(interest|prime|rate|coupon|exceed|equal to|greater than|at least|discount|premium|tax)",
    re.I,
)


@dataclass(slots=True)
class Span:
    surface: str
    start: int
    end: int


@dataclass(slots=True)
class PercentCandidate:
    value: float
    span: Span
    score: float


@dataclass(slots=True)
class ParsedFiling:
    issuer_name: Span | None = None
    cusip: Span | None = None  # span carries the source surface (round-trips)
    cusip_value: str | None = None  # normalized, check-digit-validated 9-char CUSIP
    cusip_valid: bool = False
    event_date: Span | None = None
    event_date_value: str | None = None
    percent_candidates: list[PercentCandidate] = field(default_factory=list)

    @property
    def best_percent(self) -> PercentCandidate | None:
        positives = [p for p in self.percent_candidates if p.score > 0]
        if not positives:
            return None
        return max(positives, key=lambda p: p.score)


def _find_before_label(text: str, label: str, max_back: int = 200) -> Span | None:
    idx = text.find(label)
    if idx < 0:
        return None
    window = text[max(0, idx - max_back) : idx]
    # The value is the last non-empty line before the label.
    lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
    if not lines:
        return None
    value = lines[-1]
    vstart = text.rfind(value, max(0, idx - max_back), idx)
    if vstart < 0:
        return None
    return Span(value, vstart, vstart + len(value))


def parse_filing(text: str) -> ParsedFiling:
    parsed = ParsedFiling()

    parsed.issuer_name = _find_before_label(text, "(Name of Issuer)")

    cs = extract_cusip(text)
    if cs:
        parsed.cusip = Span(cs.surface, cs.start, cs.end)
        parsed.cusip_value = cs.cusip
        parsed.cusip_valid = cs.valid

    md = _DATE_RE.search(text)
    if md:
        parsed.event_date = Span(md.group(1), md.start(1), md.end(1))
        parsed.event_date_value = md.group(1)

    # Positions of the cover-table "Percent of Class" / row-13 labels: percentages
    # near these are very likely the beneficial-ownership figure even when the
    # surrounding prose lacks ownership keywords (cover-only filings).
    row13_positions = [
        m.start()
        for m in re.finditer(r"Percent of Class Represented|Row \(1[13]\)|Percent of Class", text)
    ]

    for m in _PCT_RE.finditer(text):
        ctx = text[max(0, m.start() - 80) : m.end() + 40]
        score = 0.0
        if _POS.search(ctx):
            score += 1.0
        if _NEG.search(ctx):
            score -= 1.0
        # Proximity to a cover-table percent-of-class label.
        if any(abs(m.start() - p) <= 400 for p in row13_positions):
            score += 0.8
        # Cover-table cell: percentage isolated in whitespace, often followed by a
        # footnote marker like "(2)". Strong signal of a row-13 ownership percent.
        before = text[max(0, m.start() - 30) : m.start()]
        after = text[m.end() : m.end() + 8]
        if before.strip() == "" and re.match(r"\s*(\(\d+\))?\s*$", after):
            score += 0.7
        try:
            value = float(m.group(0).replace("%", "").strip())
        except ValueError:
            continue
        # Beneficial ownership is a fraction of a class: 0 < pct <= 100; de-rank > 100.
        if value > 100 or value == 0:
            score -= 1.0
        parsed.percent_candidates.append(
            PercentCandidate(value=value, span=Span(m.group(0), m.start(), m.end()), score=score)
        )

    return parsed
