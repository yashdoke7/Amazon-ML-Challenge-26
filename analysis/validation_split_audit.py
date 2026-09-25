"""Audit counts and positive-link ownership under the deterministic split."""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))
from validation import entity_split, f05_for_query  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
    if not root.exists():
        raise SystemExit(f"Missing supplied dataset: {root}")
    countries = {}
    with (root / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            countries[row["entity_id"]] = row["country"]
    counts = Counter()
    links = Counter()
    singleton = Counter()
    ownership = {}
    with (root / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            s1 = row["source1_entity_id"]
            bucket = entity_split(s1)
            country = countries[s1]
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            counts[bucket, country] += 1
            links[bucket, country] += len(ids)
            singleton[bucket, country] += not ids
            for target in ids:
                if target in ownership:
                    raise AssertionError(f"Target has multiple owners: {target}")
                ownership[target] = bucket
    for bucket in ("development", "validation", "training"):
        for country in sorted(set(countries.values())):
            key = (bucket, country)
            print(bucket, country, "queries", counts[key], "true_links", links[key], "singletons", singleton[key])
    print("Positive target owners:", len(ownership))
    assert f05_for_query([], []) == 1.0
    assert f05_for_query([], ["x"]) == 0.0
    assert f05_for_query(["x"], ["x"]) == 1.0
    assert f05_for_query(["x", "y"], ["x"]) == 1.25 / 1.5
    assert f05_for_query(["x"], ["x", "y"]) == 1.25 / 2.25


if __name__ == "__main__":
    main()
