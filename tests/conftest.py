"""Pytest fixtures.

Integration tests use the live Postgres database configured via
``CLEW_DATABASE_URL``. Each test runs inside a SAVEPOINT/transaction that is
rolled back, so the ledger is never polluted and tests are isolated.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from clew.db.session import get_write_engine


def _db_available() -> bool:
    try:
        with get_write_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    os.environ.get("CLEW_SKIP_DB") == "1" or not _db_available(),
    reason="live Postgres not available",
)


@pytest.fixture
def session():
    """A transactional session rolled back at the end of each test."""
    from sqlalchemy.orm import Session

    engine = get_write_engine()
    connection = engine.connect()
    trans = connection.begin()
    sess = Session(bind=connection, expire_on_commit=False)
    try:
        yield sess
    finally:
        sess.close()
        trans.rollback()
        connection.close()


_LEDGER_TABLES = "evidence, claim, mention, contradiction, entity, document, source"


@pytest.fixture
def clean_session():
    """Like ``session`` but starts from an empty ledger.

    Truncates the ledger tables *inside* the test transaction so global queries
    (supersession, contradiction detection) are isolated from any committed data.
    The rollback at teardown restores all pre-existing rows (TRUNCATE is
    transactional in Postgres), so the dev database is never disturbed.
    """
    from sqlalchemy.orm import Session

    engine = get_write_engine()
    connection = engine.connect()
    trans = connection.begin()
    connection.execute(text(f"TRUNCATE {_LEDGER_TABLES} RESTART IDENTITY CASCADE"))
    sess = Session(bind=connection, expire_on_commit=False)
    try:
        yield sess
    finally:
        sess.close()
        trans.rollback()
        connection.close()
