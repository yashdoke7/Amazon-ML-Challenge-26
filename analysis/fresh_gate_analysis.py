"""Measure which baseline match-count groups benefit from new US char address search."""

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def load(path, field, subset=None):
    out = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if subset is None or row["source1_entity_id"] in subset:
                out[row["source1_entity_id"]] = set(filter(None, (row[field] or "").split(",")))
    return out


def main():
    base = load(DEV / "tfidf_top20_fresh_word_address_tfidf_top5_fresh_name_tfidf_top10_final.tsv",
                "matched_entity_ids")
    revised = load(DEV / "tfidf_top20_fresh_word_address_tfidf_top5_fresh_name_tfidf_top10_fresh_charaddr_tfidf_top5_final.tsv",
                   "matched_entity_ids")
    truth = load(TRAIN / "train_ground_truth.tsv", "matched_entity_ids", set(base))
    stats = defaultdict(lambda: defaultdict(int))
    for q, old in base.items():
        new = revised[q]
        t = truth[q]
        k = len(old)
        for label in (str(k), "lte2" if k <= 2 else "gt2",
                      "lte3" if k <= 3 else "gt3", "all"):
            s = stats[label]
            s["queries"] += 1
            s["added_true"] += len((new - old) & t)
            s["added_false"] += len((new - old) - t)
            s["removed_true"] += len((old - new) & t)
            s["removed_false"] += len((old - new) - t)
    report = dict(stats)
    path = ROOT / "analysis/fresh_gate_analysis_results.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
