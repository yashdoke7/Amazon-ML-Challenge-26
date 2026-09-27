"""Evaluate independent retrieval ranks against a fixed labeled development split."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def parse(value):
    return [x for x in (value or "").split(",") if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", type=Path, required=True)
    parser.add_argument("--base", type=Path,
                        default=ROOT / "tmp/development_full/tfidf_top20_candidates.tsv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    with args.retrieved.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            rows[row["source1_entity_id"]] = parse(row["candidate_entity_ids"])
    ids = set(rows)
    base = {}
    with args.base.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in ids:
                base[row["source1_entity_id"]] = set(parse(row["candidate_entity_ids"]))
    truth = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in ids:
                truth[row["source1_entity_id"]] = set(parse(row["matched_entity_ids"]))
    assert ids == set(base) == set(truth)
    report = {}
    for k in (1, 5, 10, 20, 30, 50):
        standalone_true = novel_true = all_missing = 0
        macro_oracle = 0.0
        for q, ordered in rows.items():
            t = truth[q]
            c = base[q]
            r = set(ordered[:k])
            standalone_true += len(t & r)
            novel_true += len(t & r - c)
            all_missing += len(t - c)
            reachable = len(t & (c | r))
            macro_oracle += (1 if not t else
                             (1.25 * reachable / (reachable + 0.25 * len(t))
                              if reachable else 0))
        report[str(k)] = {
            "queries": len(rows), "standalone_true_links": standalone_true,
            "novel_true_links": novel_true,
            "base_missing_true_links": all_missing,
            "union_candidate_oracle": macro_oracle / len(rows),
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
