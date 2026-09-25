"""Profile true-pair name/address similarity on the first 20k shuffled labels."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
labels = []
wanted = set()
with (root / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
        if i >= 20000:
            break
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        labels.append((row["source1_entity_id"], ids))
        wanted.add(row["source1_entity_id"])
        wanted.update(ids)
records = {}
for source in (1, 2, 3):
    with (root / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in wanted:
                records[row["entity_id"]] = row

counts = Counter()
by_source = {"S2": Counter(), "S3": Counter()}
cases = []
for s1_id, ids in labels:
    s1 = records[s1_id]
    for target_id in ids:
        t = records[target_id]
        n1, n2 = s1["business_name"].casefold(), t["business_name"].casefold()
        a1, a2 = s1["business_address"].casefold(), t["business_address"].casefold()
        name = fuzz.token_sort_ratio(n1, n2)
        address = fuzz.token_sort_ratio(a1, a2) if a2 else 0
        counts["pairs"] += 1
        source_counts = by_source[target_id[:2]]
        source_counts["pairs"] += 1
        source_counts["name_below_50"] += name < 50
        source_counts["address_below_50"] += address < 50
        source_counts["missing_target_address"] += not a2
        counts["exact_name"] += n1 == n2
        counts["exact_address"] += a1 == a2
        counts["missing_target_address"] += not a2
        counts["different_country"] += s1["country"] != t["country"]
        for threshold in (30, 50, 70, 90):
            counts[f"name_below_{threshold}"] += name < threshold
            counts[f"address_below_{threshold}"] += address < threshold
        counts["both_name_address_below_50"] += name < 50 and address < 50
        if name < 30 and address >= 70 and len(cases) < 4:
            cases.append({"source1": s1, "target": t, "name_similarity": round(name, 1), "address_similarity": round(address, 1)})
print(json.dumps({"counts": counts, "rates": {k: round(v / counts["pairs"], 4) for k, v in counts.items() if k != "pairs"}, "by_source_rates": {s: {k: round(v / c["pairs"], 4) for k, v in c.items() if k != "pairs"} for s, c in by_source.items()}, "hard_examples": cases}, ensure_ascii=False, indent=2))
