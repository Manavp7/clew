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

# Percent like "19.17%", "11.0%", "9.99 %".
_PCT_RE = re.compile(r"\d{1,3}(?:\.\d{1,3})?\s*%")
_CUSIP_RE = re.compile(r"([0-9A-Z]{1,3}\s?[0-9A-Z]{2,5}\s?[0-9A-Z]{1,3})\s*\(CUSIP Number\)")
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
    cusip: Span | None = None
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

    mc = _CUSIP_RE.search(text)
    if mc:
        cusip = mc.group(1).strip()
        parsed.cusip = Span(cusip, mc.start(1), mc.start(1) + len(cusip))

    md = _DATE_RE.search(text)
    if md:
        parsed.event_date = Span(md.group(1), md.start(1), md.end(1))
        parsed.event_date_value = md.group(1)

    for m in _PCT_RE.finditer(text):
        ctx = text[max(0, m.start() - 80) : m.end() + 40]
        score = 0.0
        if _POS.search(ctx):
            score += 1.0
        if _NEG.search(ctx):
            score -= 1.0
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
