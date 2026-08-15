"""Tests for pipeline monitoring (_record logs status + duration to run_log)."""

from __future__ import annotations

from sqlalchemy import select

from clew.db.models import RunLog
from clew.pipeline.orchestrator import _record
from tests.conftest import requires_db

pytestmark = requires_db


def test_record_logs_success(clean_session):
    out = _record("teststage", lambda: {"n": 3}, session=clean_session)
    assert out["status"] == "ok" and out["n"] == 3
    row = clean_session.execute(select(RunLog)).scalars().first()
    assert row.stage == "teststage" and row.status == "ok" and row.counts == {"n": 3}


def test_record_logs_error_and_continues(clean_session):
    def boom():
        raise RuntimeError("kaboom")

    out = _record("badstage", boom, session=clean_session)
    assert out["status"] == "error"
    row = clean_session.execute(select(RunLog)).scalars().first()
    assert row.status == "error" and "kaboom" in row.error
