"""Name normalization for entity resolution.

``normalized``: lowercased, punctuation-stripped, whitespace-collapsed form used
for similarity scoring. ``core``: ``normalized`` minus common corporate suffixes,
used as a blocking key so only plausibly-matching records are compared.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[.,/&()'\"-]")
_WS = re.compile(r"\s+")

# Corporate / legal suffixes stripped to form the blocking 'core' name.
_SUFFIXES = {
    "inc", "incorporated", "llc", "l l c", "lp", "l p", "llp", "ltd", "limited",
    "corp", "corporation", "co", "company", "plc", "nv", "sa", "ag", "gmbh",
    "trust", "fund", "partners", "capital", "management", "group", "holdings",
    "the", "and",
}


def normalized(name: str) -> str:
    s = name.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def core(name: str) -> str:
    toks = [t for t in normalized(name).split(" ") if t and t not in _SUFFIXES]
    return " ".join(toks) if toks else normalized(name)


def block_key(name: str) -> str:
    """First token of the core name (cheap, high-recall blocking key)."""
    c = core(name)
    return c.split(" ")[0][:6] if c else ""
