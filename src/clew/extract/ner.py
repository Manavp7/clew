"""GLiNER-based mention detection (CPU, zero-shot).

Produces surface mentions with exact character offsets into ``document.text``.
Long filings are chunked with offsets adjusted back to document coordinates so
every mention round-trips: ``text[start:end] == surface``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from clew.packs.pack_a_financial import NER_LABELS

GLINER_MODEL = "urchade/gliner_small-v2.1"
GLINER_VERSION = f"gliner@{GLINER_MODEL}"

# Map GLiNER zero-shot labels -> normalized ner_type stored on the mention.
_LABEL_MAP = {
    "organization": "Organization",
    "person": "Person",
    "percentage": "PERCENT",
    "monetary amount": "MONEY",
    "date": "DATE",
    "security class": "SECURITY",
}


# Generic boilerplate surfaces that are not real, resolvable entities.
_STOPLIST = {
    "sec", "securities and exchange commission", "securities & exchange commission",
    "the issuer", "issuer", "the company", "company", "reporting person",
    "reporting persons", "the reporting person", "the reporting persons",
    "filing person", "filing persons", "common stock", "ordinary shares",
    "schedule 13d", "schedule 13g", "washington", "d.c.", "united states",
    "the securities", "the shares", "the commission",
}


def _is_noise(surface: str, ner_type: str) -> bool:
    s = surface.strip().lower()
    if ner_type in ("Organization", "Person"):
        if s in _STOPLIST or len(s) < 3:
            return True
    return False


@dataclass(slots=True)
class MentionSpan:
    surface: str
    start: int
    end: int
    ner_type: str
    score: float


@lru_cache
def _model():
    from gliner import GLiNER

    return GLiNER.from_pretrained(GLINER_MODEL)


def _chunks(text: str, size: int = 1200, overlap: int = 150):
    """Yield (chunk_text, base_offset) windows so offsets map back to the document."""
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + size, n)
        yield text[pos:end], pos
        if end == n:
            break
        pos = end - overlap


def extract_mentions(
    text: str, *, labels: list[str] | None = None, threshold: float = 0.5
) -> list[MentionSpan]:
    model = _model()
    labels = labels or NER_LABELS
    seen: set[tuple[int, int]] = set()
    out: list[MentionSpan] = []

    for chunk, base in _chunks(text):
        for ent in model.predict_entities(chunk, labels, threshold=threshold):
            start = base + ent["start"]
            end = base + ent["end"]
            if (start, end) in seen:
                continue
            seen.add((start, end))
            surface = text[start:end]
            if surface != ent["text"]:
                # offset drift guard: skip rather than store a non-round-tripping span
                continue
            ner_type = _LABEL_MAP.get(ent["label"], ent["label"])
            if _is_noise(surface, ner_type):
                continue
            out.append(
                MentionSpan(
                    surface=surface,
                    start=start,
                    end=end,
                    ner_type=ner_type,
                    score=float(ent["score"]),
                )
            )
    out.sort(key=lambda m: m.start)
    return out
