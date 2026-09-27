"""Paired query bootstrap for a measured macro-F0.5 revision."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRUTH = (ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
         / "train_ground_truth.tsv")


def load(path, key, field, subset=None):
    rows = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if subset is None or row[key] in subset:
                rows[row[key]] = set(filter(None, (row[field] or "").split(",")))
    return rows


def f05(pred, truth):
    if not pred and not truth:
        return 1.0
    return 1.25 * len(pred & truth) / (len(pred) + 0.25 * len(truth)) if pred else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = load(args.base, "source1_entity_id", "matched_entity_ids")
    new = load(args.new, "source1_entity_id", "matched_entity_ids")
    truth = load(TRUTH, "source1_entity_id", "matched_entity_ids", set(base))
    assert base.keys() == new.keys() == truth.keys()
    delta = np.asarray([f05(new[q], truth[q]) - f05(base[q], truth[q]) for q in base],
                       dtype=np.float64)
    rng = np.random.default_rng(260927)
    draws = np.empty(5000, dtype=np.float64)
    for i in range(len(draws)):
        draws[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    report = {"queries": len(delta), "mean_gain": float(delta.mean()),
              "queries_improved": int((delta > 0).sum()),
              "queries_worsened": int((delta < 0).sum()),
              "bootstrap_95_pct_interval": [float(x) for x in np.quantile(draws, [0.025, 0.975])]}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
