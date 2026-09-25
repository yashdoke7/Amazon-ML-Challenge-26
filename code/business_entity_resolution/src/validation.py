"""Deterministic entity split and the challenge's per-Source-1 macro F0.5.

Keep all positive targets of a Source-1 owner in that owner's split. During
training, exclude targets owned by development/validation Source-1 rows from
negative examples, even when retrieval used the full target corpus.
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def entity_split(source1_id: str, seed: str = "amazon-ml-2026-v1") -> str:
    """Stable approximate 1% development, 4% validation, 95% training split."""
    digest = hashlib.blake2b(f"{seed}:{source1_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 10_000
    if bucket < 100:
        return "development"
    if bucket < 500:
        return "validation"
    return "training"


def f05_for_query(truth: Iterable[str], predicted: Iterable[str]) -> float:
    true_ids, pred_ids = set(truth), set(predicted)
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn) if tp else 0.0


def macro_f05(truth_by_s1: dict[str, set[str]], pred_by_s1: dict[str, set[str]]) -> float:
    """Score exactly one prediction set per truth S1, including empty lists."""
    if set(truth_by_s1) != set(pred_by_s1):
        missing = len(set(truth_by_s1) - set(pred_by_s1))
        extra = len(set(pred_by_s1) - set(truth_by_s1))
        raise ValueError(f"Source-1 ID mismatch: {missing} missing, {extra} extra")
    if not truth_by_s1:
        raise ValueError("Cannot score an empty validation split")
    return sum(f05_for_query(truth_by_s1[k], pred_by_s1[k]) for k in truth_by_s1) / len(truth_by_s1)
