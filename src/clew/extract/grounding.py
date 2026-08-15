"""Ground an extracted evidence quote to exact char offsets in the document.

Guarantees the stored snippet round-trips (``text[start:end] == snippet``):

1. exact substring match (fast path),
2. whitespace-normalized exact match,
3. fuzzy alignment (rapidfuzz) as a last resort.

If no span scores above threshold, the quote is rejected — keeping the
"every claim has a real evidence span" invariant honest.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

_WS = re.compile(r"\s+")


def ground_quote(text: str, quote: str, *, min_score: float = 85.0) -> tuple[int, int, str] | None:
    quote = quote.strip()
    if not quote:
        return None

    # 1) exact
    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote), quote

    # 2) whitespace-normalized exact: map normalized match back to raw offsets
    norm_quote = _WS.sub(" ", quote)
    span = _normalized_find(text, norm_quote)
    if span is not None:
        return span

    # 3) fuzzy sliding window around the best anchor token
    return _fuzzy_find(text, quote, min_score)


def _normalized_find(text: str, norm_quote: str) -> tuple[int, int, str] | None:
    # Build a regex that allows flexible whitespace between the quote's tokens.
    tokens = [re.escape(t) for t in norm_quote.split(" ") if t]
    if not tokens:
        return None
    pattern = r"\s+".join(tokens)
    m = re.search(pattern, text)
    if m:
        return m.start(), m.end(), text[m.start() : m.end()]
    return None


def _fuzzy_find(text: str, quote: str, min_score: float) -> tuple[int, int, str] | None:
    qlen = len(quote)
    best = (min_score, -1, -1)
    step = max(1, qlen // 4)
    # Scan windows of roughly the quote length.
    for start in range(0, max(1, len(text) - qlen + 1), step):
        window = text[start : start + qlen]
        score = fuzz.ratio(window, quote)
        if score > best[0]:
            best = (score, start, start + qlen)
    if best[1] < 0:
        return None
    s, e = best[1], best[2]
    # Trim to word boundaries for a cleaner snippet.
    return s, e, text[s:e]
