"""Incremental pipeline orchestration + monitoring.

Runs the processing stages in order, each one **incremental** (skips
already-processed documents via the stage guards) and **logged** to ``run_log``
with counts + duration + git sha. This is the single-machine alternative to a
durable workflow engine (Temporal is deferred); re-running only does new work.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

from clew.db.models import RunLog
from clew.db.session import write_session


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _record(stage: str, fn: Callable[[], dict], session=None) -> dict:
    sha = _git_sha()
    t0 = time.monotonic()
    status, error, counts = "ok", None, {}
    try:
        counts = fn() or {}
    except Exception as exc:  # noqa: BLE001 - log + continue to next stage
        status, error = "error", str(exc)
    dur = int((time.monotonic() - t0) * 1000)
    row = RunLog(
        stage=stage, status=status, counts=counts, duration_ms=dur,
        pipeline_git_sha=sha, error=error,
    )
    if session is not None:
        session.add(row)
        session.flush()
    else:
        with write_session() as own:
            own.add(row)
    return {"stage": stage, "status": status, "duration_ms": dur, **counts}


def run_pipeline(pack: str = "all") -> dict:
    """Run the incremental processing chain. ``pack``: a | b | all.

    Assumes documents are already ingested (``clew ingest ...``). Each stage is
    idempotent/incremental, so this is safe to re-run after new ingests.
    """
    from clew.extract.service import run_claims, run_mentions
    from clew.ledger.contradiction import materialize_contradictions
    from clew.ledger.supersession import reconcile_supersessions
    from clew.project.graph import build_graph, graph_to_json
    from clew.project.vectors import build_entity_vectors
    from clew.resolve.service import run_resolution

    steps: list[tuple[str, Callable[[], dict]]] = []
    if pack in ("a", "all"):
        steps += [
            ("mentions", lambda: run_mentions()),
            ("resolve", lambda: run_resolution()),
            ("claims", lambda: run_claims()),
        ]
    if pack in ("b", "all"):
        from clew.extract.science import run_science_claims

        steps.append(("science", lambda: run_science_claims()))
    steps += [
        ("supersede", _reconcile_supersede(reconcile_supersessions)),
        ("contradictions", _materialize(materialize_contradictions)),
        ("vectors", lambda: build_entity_vectors(only_missing=True)),
        ("graph", _project_graph(build_graph, graph_to_json)),
    ]

    results = [_record(stage, fn) for stage, fn in steps]
    return {"pipeline": pack, "stages": results}


def _reconcile_supersede(fn):
    def run():
        with write_session() as s:
            return fn(s)

    return run


def _materialize(fn):
    def run():
        with write_session() as s:
            return fn(s)

    return run


def _project_graph(build_graph, graph_to_json):
    def run():
        import json
        from pathlib import Path

        from clew.db.session import read_session

        with read_session() as s:
            g = build_graph(s)
            data = graph_to_json(g)
        Path("data").mkdir(exist_ok=True)
        Path("data/graph.json").write_text(json.dumps(data))
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}

    return run
