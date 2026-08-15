"""Test the PyMuPDF PDF parsing backend (offset-stable text)."""

from __future__ import annotations

import os
import tempfile

import pytest

fitz = pytest.importorskip("fitz")

from clew.parse.pdf import parse_pdf  # noqa: E402


def test_parse_pdf_pymupdf_roundtrip():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Acme Capital LP beneficially owns 7.5% of Target Corp.")
    path = os.path.join(tempfile.gettempdir(), "clew_test_parse.pdf")
    doc.save(path)
    doc.close()

    text = parse_pdf(path, backend="pymupdf")
    assert "Acme Capital LP" in text
    assert "7.5%" in text
    # Offsets into the returned string round-trip (same invariant as HTML path).
    idx = text.index("7.5%")
    assert text[idx : idx + 4] == "7.5%"
