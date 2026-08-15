"""Stage metrics for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def pairwise_er_metrics(
    predicted_same: set[frozenset], gold_same: set[frozenset], gold_diff: set[frozenset]
) -> dict:
    """Compare predicted merges against labeled same/different pairs.

    * false_merge_rate = predicted-same among gold-different
    * missed_merge_rate = predicted-NOT-same among gold-same
    """
    fm = sum(1 for p in gold_diff if p in predicted_same)
    mm = sum(1 for p in gold_same if p not in predicted_same)
    n_diff = len(gold_diff) or 1
    n_same = len(gold_same) or 1
    correct_same = len(gold_same) - mm
    correct_diff = len(gold_diff) - fm
    total = len(gold_same) + len(gold_diff)
    return {
        "false_merge_rate": round(fm / n_diff, 4),
        "missed_merge_rate": round(mm / n_same, 4),
        "pair_accuracy": round((correct_same + correct_diff) / total, 4) if total else 0.0,
        "gold_same": len(gold_same),
        "gold_diff": len(gold_diff),
    }
