"""Persist raw documents into the ledger: dedupe + raw storage + document rows."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from clew.config import get_settings
from clew.db.models import Document, Source
from clew.db.session import write_session
from clew.ingest.base import Connector, RawDocument


def _get_or_create_source(session, name: str) -> Source:
    src = session.execute(select(Source).where(Source.name == name)).scalars().first()
    if src is None:
        src = Source(name=name, url="https://www.sec.gov/edgar", publisher="U.S. SEC")
        session.add(src)
        session.flush()
    return src


def _persist_one(session, src: Source, raw: RawDocument, raw_dir: Path) -> Document | None:
    content_hash = raw.content_hash()
    existing = session.execute(
        select(Document).where(
            Document.source_id == src.id, Document.content_hash == content_hash
        )
    ).scalars().first()
    if existing is not None:
        return None  # dedupe

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{raw.external_id.replace('/', '_')}.txt"
    raw_path.write_text(raw.text, encoding="utf-8")

    doc = Document(
        source_id=src.id,
        external_id=raw.external_id,
        doc_type=raw.doc_type,
        url=raw.url,
        retrieved_at=raw.retrieved_at,
        content_hash=content_hash,
        raw_path=str(raw_path),
        text=raw.text,
        meta=raw.meta,
    )
    session.add(doc)
    session.flush()
    return doc


def ingest_documents(
    connector: Connector, target: int, *, fetch_cap: int | None = None, **fetch_kwargs
) -> dict:
    """Fetch via ``connector`` and persist until ``target`` *distinct* docs land.

    Identical-content filings (EDGAR joint/cross-filings) are deduped by
    ``content_hash``; we keep consuming the source until ``target`` new documents
    are stored or the source is exhausted (bounded by ``fetch_cap``).
    """
    settings = get_settings()
    raw_dir = settings.raw_dir / connector.source_name.replace(" ", "_")
    cap = fetch_cap or target * 4
    ingested, skipped = 0, 0

    with write_session() as session:
        src = _get_or_create_source(session, connector.source_name)
        for raw in connector.fetch(cap, **fetch_kwargs):
            doc = _persist_one(session, src, raw, raw_dir)
            if doc is None:
                skipped += 1
            else:
                ingested += 1
            if ingested >= target:
                break
    return {"ingested": ingested, "skipped_duplicates": skipped, "source": connector.source_name}
