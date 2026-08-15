"""PDF parsing → offset-stable normalized text.

Default backend is **PyMuPDF** (fast, CPU-only). **MinerU** is an optional,
higher-fidelity backend for table-heavy filings (lazy-imported; install via the
``pdf-mineru`` extra). Both return a single normalized string so downstream char
offsets index into it consistently — the same invariant the HTML/XBRL path uses.

Backend is selected by ``CLEW_PDF_BACKEND`` (``pymupdf`` | ``mineru``).
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(path: str | Path, backend: str | None = None) -> str:
    backend = backend or _default_backend()
    if backend == "mineru":
        return _parse_with_mineru(Path(path))
    return _parse_with_pymupdf(Path(path))


def _default_backend() -> str:
    import os

    return os.environ.get("CLEW_PDF_BACKEND", "pymupdf").lower()


def _parse_with_pymupdf(path: Path) -> str:
    import fitz  # PyMuPDF

    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    # Join pages with a form feed so page boundaries are recoverable but offsets
    # remain a single coherent string.
    return "\f".join(parts)


def _parse_with_mineru(path: Path) -> str:
    """Optional MinerU backend (lazy import; only if the extra is installed)."""
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe  # type: ignore
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MinerU backend requested but not installed. Install with "
            "`uv sync --extra pdf-mineru`, or use CLEW_PDF_BACKEND=pymupdf."
        ) from exc

    pdf_bytes = path.read_bytes()
    writer = DiskReaderWriter(str(path.parent))
    pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    return pipe.pipe_mk_markdown(str(path.parent), drop_mode="none")
