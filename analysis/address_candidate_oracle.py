"""Measure the upper bound of the selected top-40 plus address quota."""

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from number_veto_full_development import score  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
BASE = ROOT / "tmp" / "development_full" / "generalized_top40_candidates.tsv"
EXTRA = ROOT / "analysis" / "india_address_tfidf_dev_candidates.tsv"


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {r["source1_entity_id"]: r["candidate_entity_ids"].split(",")
                if r["candidate_entity_ids"] else []
                for r in csv.DictReader(stream, delimiter="\t")}


def main():
    base = read(BASE)
    extra = read(EXTRA)
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in base:
                truth[q] = set(row["matched_entity_ids"].split(",")) if row[
                    "matched_entity_ids"] else set()
    countries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in base:
                countries[row["entity_id"]] = row["country"]
    assert set(base) == set(truth) == set(countries)
    results = {}
    for quota in (0, 1, 5, 10, 20, 50):
        retrieved = {q: (set(base[q]) | set(extra.get(q, [])[:quota])) & truth[q]
                     for q in base}
        results[str(quota)] = score(retrieved, truth, countries)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
