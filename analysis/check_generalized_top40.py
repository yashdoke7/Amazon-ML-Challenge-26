"""Measure the exact labeled cost of compact blocking for the new matcher."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from validation import f05_for_query  # noqa: E402

TEMP = ROOT / "tmp" / "development_full"
TRUTH = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"


def load(path, id_column):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[id_column].split(","))
                if row[id_column] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def metric(groups, truth):
    return {"macro_f05": round(sum(f05_for_query(truth[q], groups[q]) for q in truth) / len(truth), 6),
            "tp": sum(len(groups[q] & truth[q]) for q in truth),
            "fp": sum(len(groups[q] - truth[q]) for q in truth)}


def main():
    full = load(TEMP / "generalized_results.tsv", "matched_entity_ids")
    compact = load(TEMP / "generalized_top40_results.tsv", "matched_entity_ids")
    candidates = load(TEMP / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    with TRUTH.open(encoding="utf-8", newline="") as stream:
        truth = {q: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
                 for row in csv.DictReader(stream, delimiter="\t")
                 if (q := row["source1_entity_id"]) in full}
    assert set(full) == set(compact) == set(candidates) == set(truth)
    assert all(compact[q] <= candidates[q] for q in truth)
    result = {"queries": len(truth), "candidate_pairs": sum(map(len, candidates.values())),
              "candidate_true_links": sum(len(candidates[q] & truth[q]) for q in truth),
              "full": metric(full, truth), "compact": metric(compact, truth),
              "lost_selected_links": sum(len(full[q] - compact[q]) for q in truth),
              "lost_selected_true": sum(len((full[q] - compact[q]) & truth[q]) for q in truth),
              "lost_selected_false": sum(len((full[q] - compact[q]) - truth[q]) for q in truth)}
    (ROOT / "analysis" / "generalized_top40_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
