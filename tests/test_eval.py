"""Unit tests for evaluation metrics (pure functions, no DB)."""

from __future__ import annotations

from clew.eval.metrics import pairwise_er_metrics, prf


def test_prf():
    m = prf(tp=3, fp=1, fn=1)
    assert m.precision == 0.75
    assert m.recall == 0.75
    assert round(m.f1, 2) == 0.75


def test_prf_empty():
    m = prf(0, 0, 0)
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


def test_pairwise_er_metrics():
    p1, p2, p3 = frozenset({"a", "b"}), frozenset({"c", "d"}), frozenset({"e", "f"})
    gold_same = {p1, p2}
    gold_diff = {p3}
    predicted_same = {p1, p3}  # p2 missed (same not merged), p3 false-merged
    m = pairwise_er_metrics(predicted_same, gold_same, gold_diff)
    assert m["missed_merge_rate"] == 0.5  # 1 of 2 same pairs missed
    assert m["false_merge_rate"] == 1.0  # 1 of 1 diff pair merged
