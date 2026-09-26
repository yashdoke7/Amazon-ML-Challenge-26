"""Account for every development false negative after the Unicode matcher."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"


def read_ids(path, column):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[column].split(","))
                if row[column] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    selected = read_ids(DEV / "generalized_exclusive.tsv", "matched_entity_ids")
    candidates = read_ids(DEV / "candidate_pairs.tsv", "candidate_entity_ids")
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in selected:
                truth[q] = set(row["matched_entity_ids"].split(",")) \
                    if row["matched_entity_ids"] else set()
    country = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in selected:
                country[row["entity_id"]] = row["country"]
    assert set(selected) == set(candidates) == set(truth) == set(country)
    links = Counter()
    groups = Counter()
    for q, true_ids in truth.items():
        c, p = candidates[q], selected[q]
        region = country[q]
        groups[region, "queries"] += 1
        groups[region, "true_empty"] += not true_ids
        groups[region, "predicted_empty"] += not p
        groups[region, "complete"] += p == true_ids
        groups[region, "all_true_retrieved"] += true_ids <= c
        groups[region, "has_anchor_and_missed"] += bool(p & true_ids) and bool(true_ids - p)
        groups[region, "all_true_missed"] += bool(true_ids) and not (p & true_ids)
        for t in true_ids | p:
            source = t.split("-", 1)[0]
            key = (region, source)
            if t in true_ids and t in p:
                links[key, "tp"] += 1
            elif t in true_ids and t not in c:
                links[key, "missed_retrieval"] += 1
            elif t in true_ids:
                links[key, "rejected_matcher"] += 1
            else:
                links[key, "fp"] += 1
    result = {"queries": len(truth),
              "groups": {region: {kind: value for (r, kind), value in groups.items() if r == region}
                         for region in sorted(set(country.values()))},
              "links": {f"{region}/{source}":
                        {kind: value for ((r, s), kind), value in links.items()
                         if r == region and s == source}
                        for region, source in sorted(set(key for key, _ in links))}}
    (ROOT / "analysis" / "generalized_gap_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
