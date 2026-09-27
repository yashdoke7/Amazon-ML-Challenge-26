"""Positive-only upper bound for rescue through an already selected true sibling.

This is diagnostic only: inference does not know which selected links are true.
"""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def read(path, field, quota=None):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")[:quota])
                if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    base = read(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    address = read(ROOT / "analysis/india_address_tfidf_dev_candidates.tsv",
                   "candidate_entity_ids", 20)
    selected = read(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    truth = read(DATA / "train_ground_truth.tsv", "matched_entity_ids")
    counts = Counter()
    for query, base_ids in base.items():
        truth_ids = truth[query]
        true_selected = selected[query] & truth_ids
        pool = base_ids | address.get(query, set())
        for target in truth_ids - selected[query]:
            stage = "rejected" if target in pool else "missing"
            counts[(stage, "total")] += 1
            if true_selected:
                counts[(stage, "has_true_selected_sibling")] += 1
                if any(sibling.split("-", 1)[0] != target.split("-", 1)[0]
                       for sibling in true_selected):
                    counts[(stage, "has_true_selected_other_source")] += 1
        if truth_ids - selected[query]:
            counts[("queries_with_missed_links", "total")] += 1
            if true_selected:
                counts[("queries_with_missed_links", "has_true_selected_sibling")] += 1
    result = {"|".join(key): value for key, value in sorted(counts.items())}
    dest = ROOT / "analysis/sibling_link_headroom_results.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
