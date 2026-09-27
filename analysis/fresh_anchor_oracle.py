"""Optimistic upper bound if one retrieved true target revealed its full group.

Uses labels only for analysis, never for retrieval or submission.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv"


def values(text):
    return set(filter(None, (text or "").split(",")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--extra", type=Path)
    p.add_argument("--extra-top-k", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    candidates = {}
    with args.base.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            candidates[r["source1_entity_id"]] = values(r["candidate_entity_ids"])
    if args.extra:
        with args.extra.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                candidates[r["source1_entity_id"]].update(
                    (r["candidate_entity_ids"] or "").split(",")[:args.extra_top_k]
                )
    total = empty = no_anchor = partial = complete = 0
    missed_links_with_anchor = missed_links_without_anchor = 0
    no_anchor_sizes = Counter()
    with TRUTH.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            q = r["source1_entity_id"]
            if q not in candidates:
                continue
            total += 1
            truth = values(r["matched_entity_ids"])
            if not truth:
                empty += 1
                continue
            found = len(truth & candidates[q])
            if found == 0:
                no_anchor += 1
                missed_links_without_anchor += len(truth)
                no_anchor_sizes[len(truth)] += 1
            elif found == len(truth):
                complete += 1
            else:
                partial += 1
                missed_links_with_anchor += len(truth) - found
    report = {
        "queries": total, "true_empty_groups": empty,
        "nonempty_groups_with_zero_retrieved_true": no_anchor,
        "nonempty_groups_partially_retrieved": partial,
        "nonempty_groups_fully_retrieved": complete,
        "missing_true_links_in_partially_retrieved_groups": missed_links_with_anchor,
        "missing_true_links_in_zero_anchor_groups": missed_links_without_anchor,
        "zero_anchor_group_sizes": dict(sorted(no_anchor_sizes.items())),
        "perfect_group_propagation_oracle": 1.0 - no_anchor / total,
        "meaning": "Upper bound assuming a perfect target-target cluster and perfect decisions, not a feasible achieved score",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
