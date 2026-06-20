"""Evaluation harness — versioned metrics over the pipeline.

North-star metrics: ER accuracy and citation accuracy. Every run is recorded in
the ``eval_run`` table with the pipeline git SHA and model versions, so quality
is tracked over time against the exact code/models that produced it.
"""

from __future__ import annotations

import re
import subprocess

from sqlalchemy import select

from clew.db.models import Claim, Document, EvalRun, Evidence
from clew.db.session import read_session, write_session
from clew.eval.datasets import load_er_pairs, load_extraction_gold
from clew.eval.metrics import pairwise_er_metrics, prf
from clew.extract.ner import GLINER_VERSION
from clew.resolve.resolver import Record, resolve_records


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _model_versions() -> dict:
    from clew.config import get_settings

    s = get_settings()
    return {
        "ner": GLINER_VERSION,
        "extraction_model": s.extraction_model if s.has_llm else "rule@0.1",
        "embedding_model": s.embedding_model,
    }


def _persist(stage: str, dataset: str, metrics: dict) -> None:
    with write_session() as session:
        session.add(
            EvalRun(
                stage=stage,
                dataset=dataset,
                pipeline_git_sha=_git_sha(),
                model_versions=_model_versions(),
                metrics=metrics,
            )
        )


def eval_er() -> dict:
    """Resolve labeled name pairs and score false/missed-merge rates."""
    gold = load_er_pairs()
    gold_same: set[frozenset] = set()
    gold_diff: set[frozenset] = set()
    predicted_same: set[frozenset] = set()

    for i, pair in enumerate(gold["pairs"]):
        a, b = pair["a"], pair["b"]
        key = frozenset({f"{i}:a", f"{i}:b"})
        (gold_same if pair["label"] == "same" else gold_diff).add(key)
        # Resolve this isolated pair; same cluster => predicted same.
        records = [
            Record(0, a, "Organization", mention_id=0),
            Record(1, b, "Organization", mention_id=1),
        ]
        clusters = resolve_records(records)
        if len(clusters) == 1:
            predicted_same.add(key)

    metrics = pairwise_er_metrics(predicted_same, gold_same, gold_diff)
    _persist("er", gold["dataset"], metrics)
    return metrics


def eval_extraction() -> dict:
    """Compare written OWNS claims against per-filing gold relationships."""
    gold = load_extraction_gold()
    expected = {
        f["accession"]: (f["filer_cik"], f["subject_cik"], f["predicate"]) for f in gold["filings"]
    }
    tp = fp = fn = 0
    with read_session() as session:
        for accession, (filer_cik, subject_cik, predicate) in expected.items():
            doc = session.execute(
                select(Document).where(Document.external_id == accession)
            ).scalars().first()
            if doc is None:
                fn += 1
                continue
            claims = _claims_for_doc(session, doc.id)
            match = any(
                c.predicate == predicate
                and _cik(session, c.subject_id) == filer_cik
                and _cik(session, c.object_id) == subject_cik
                for c in claims
            )
            if match:
                tp += 1
            else:
                fn += 1

    metrics = prf(tp, fp, fn).as_dict()
    metrics["note"] = "relationship-level recall vs per-filing gold"
    _persist("extraction", gold["dataset"], metrics)
    return metrics


def eval_citation() -> dict:
    """Citation accuracy: does each evidence span support its claim?

    For ownership claims with a ``stake_pct`` qualifier we require the percent
    value to appear in the evidence snippet. All evidence must round-trip to the
    document text (already enforced at write time; re-verified here).
    """
    checked = supported = roundtrip_ok = 0
    with read_session() as session:
        claims = session.execute(select(Claim)).scalars().all()
        for c in claims:
            ev = session.execute(
                select(Evidence).where(Evidence.claim_id == c.id)
            ).scalars().first()
            if ev is None:
                continue
            checked += 1
            doc = session.get(Document, ev.document_id)
            if doc and doc.text[ev.char_start : ev.char_end] == ev.snippet:
                roundtrip_ok += 1
            stake = (c.qualifiers or {}).get("stake_pct")
            if stake is None:
                supported += 1  # no numeric claim to contradict; span is provenance
            else:
                nums = {float(x) for x in re.findall(r"\d+\.?\d*", ev.snippet)}
                supported += 1 if float(stake) in nums else 0

    metrics = {
        "claims_checked": checked,
        "citation_accuracy": round(supported / checked, 4) if checked else 0.0,
        "evidence_roundtrip_rate": round(roundtrip_ok / checked, 4) if checked else 0.0,
    }
    _persist("reasoning", "citation@v1", metrics)
    return metrics


def _claims_for_doc(session, doc_id: int) -> list[Claim]:
    claim_ids = session.execute(
        select(Evidence.claim_id).where(Evidence.document_id == doc_id)
    ).scalars().all()
    if not claim_ids:
        return []
    return list(
        session.execute(select(Claim).where(Claim.id.in_(set(claim_ids)))).scalars().all()
    )


def _cik(session, entity_id: str | None) -> str | None:
    if not entity_id:
        return None
    from clew.db.models import Entity

    e = session.get(Entity, entity_id)
    return (e.external_ids or {}).get("cik") if e else None


def _score_backend_on_gold(backend: str) -> dict:
    """Resolve the full ER-gold record set with ``backend`` and score pairwise."""
    from clew.resolve.resolver import Record, resolve_records

    gold = load_er_pairs()
    records: list[Record] = []
    pair_rids: list[tuple[int, int, str]] = []
    rid = 0
    for pair in gold["pairs"]:
        a_rid, b_rid = rid, rid + 1
        records.append(Record(a_rid, pair["a"], "Organization"))
        records.append(Record(b_rid, pair["b"], "Organization"))
        pair_rids.append((a_rid, b_rid, pair["label"]))
        rid += 2

    if backend == "splink":
        from clew.resolve.splink_er import resolve_records_splink

        clusters = resolve_records_splink(records, train=False)
    else:
        clusters = resolve_records(records)

    rid_to_cluster: dict[int, int] = {}
    for ci, c in enumerate(clusters):
        for m in c.members:
            rid_to_cluster[m.rid] = ci

    gold_same, gold_diff, predicted_same = set(), set(), set()
    for a_rid, b_rid, label in pair_rids:
        key = frozenset({a_rid, b_rid})
        (gold_same if label == "same" else gold_diff).add(key)
        if rid_to_cluster.get(a_rid) == rid_to_cluster.get(b_rid):
            predicted_same.add(key)
    return pairwise_er_metrics(predicted_same, gold_same, gold_diff)


def eval_er_compare() -> dict:
    """Compare ER backends on the gold set; recommend the default by metrics.

    Promotes Splink to default ONLY if it strictly beats the union-find resolver
    on pair-accuracy (then false-merge rate). Otherwise keeps the union-find
    default. Persists a versioned eval_run for each backend.
    """
    results: dict[str, dict] = {}
    for backend in ("default", "splink"):
        try:
            m = _score_backend_on_gold(backend)
        except Exception as exc:  # noqa: BLE001 - splink optional/fragile
            m = {"error": str(exc)}
        results[backend] = m
        _persist("er", f"er_pairs@v1:{backend}", m)

    d, s = results["default"], results.get("splink", {})

    def _key(m: dict) -> tuple[float, float]:
        if "error" in m:
            return (-1.0, -1.0)
        return (m.get("pair_accuracy", 0.0), -m.get("false_merge_rate", 1.0))

    recommended = "splink" if _key(s) > _key(d) else "default"
    return {"results": results, "recommended_default": recommended}


def eval_science() -> dict:
    """Pack-B extraction check: known works have expected authorship + citations.

    Skips gracefully (status='skipped') if no OpenAlex works are ingested, so the
    harness is safe to run on a Pack-A-only ledger.
    """
    from clew.db.models import Document
    from clew.eval.datasets import load_science_gold

    gold = load_science_gold()
    checked = author_ok = cites_ok = 0
    with read_session() as session:
        for w in gold["works"]:
            doc = session.execute(
                select(Document).where(Document.external_id == w["openalex"])
            ).scalars().first()
            if doc is None:
                continue
            checked += 1
            claim_ids = session.execute(
                select(Evidence.claim_id).where(Evidence.document_id == doc.id)
            ).scalars().all()
            claims = list(
                session.execute(select(Claim).where(Claim.id.in_(set(claim_ids)))).scalars().all()
            )
            authored = [c for c in claims if c.predicate == "AUTHORED"]
            cites = [c for c in claims if c.predicate == "CITES"]
            from clew.db.models import Entity

            names = {
                (session.get(Entity, c.subject_id).canonical_name) for c in authored
            }
            if w["expect_author_name"] in names:
                author_ok += 1
            if len(cites) >= w.get("min_citations", 1):
                cites_ok += 1

    if checked == 0:
        metrics = {"status": "skipped", "reason": "no OpenAlex works ingested"}
    else:
        metrics = {
            "works_checked": checked,
            "author_recall": round(author_ok / checked, 4),
            "citation_coverage": round(cites_ok / checked, 4),
        }
    _persist("extraction", "science_openalex@v1", metrics)
    return metrics


def eval_all() -> dict:
    return {
        "er": eval_er(),
        "extraction": eval_extraction(),
        "citation": eval_citation(),
        "science": eval_science(),
        "er_backend_comparison": eval_er_compare(),
    }
