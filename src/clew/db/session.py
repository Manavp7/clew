"""Database engine/session management with read/write role separation.

Architectural constraint: **only extraction pipelines may create or update
claims.** We enforce this at the session layer:

* :func:`write_session` — used by ingestion/extraction/ER/projection writers.
* :func:`read_session` — used by the reasoning/API layer. It opens a
  ``READ ONLY`` transaction so that *any* attempt to write raises at the DB
  level, regardless of which role is configured. If ``CLEW_DATABASE_URL_RO`` is
  set it additionally connects as a dedicated read-only role.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from clew.config import get_settings


@lru_cache
def get_write_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache
def get_read_engine() -> Engine:
    settings = get_settings()
    dsn = settings.database_url_ro or settings.database_url
    return create_engine(dsn, pool_pre_ping=True, future=True)


_WriteSession = sessionmaker(bind=None, class_=Session, expire_on_commit=False, future=True)
_ReadSession = sessionmaker(bind=None, class_=Session, expire_on_commit=False, future=True)


@contextmanager
def write_session() -> Iterator[Session]:
    """Read/write session for extraction pipelines. Commits on success."""
    session = _WriteSession(bind=get_write_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def read_session() -> Iterator[Session]:
    """Read-only session for the reasoning/API layer.

    Opens a ``READ ONLY`` transaction so writes are rejected by Postgres even if
    a write-capable role is configured. This is the enforcement point for the
    "reasoning layer cannot write claims" constraint.
    """
    session = _ReadSession(bind=get_read_engine())
    try:
        session.execute(text("SET TRANSACTION READ ONLY"))
        yield session
    finally:
        session.rollback()
        session.close()
