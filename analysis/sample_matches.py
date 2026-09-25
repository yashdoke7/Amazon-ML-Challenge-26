"""Print a small, reproducible set of true-match examples from supplied data."""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"

chosen = []
with (root / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        n = len(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else 0
        if len(chosen) < 3 or (n == 0 and not any(x["kind"] == "singleton" for x in chosen)) or (n == 1 and not any(x["kind"] == "one" for x in chosen)):
            chosen.append({**row, "kind": "singleton" if n == 0 else "one" if n == 1 else "many"})
        if len(chosen) == 5:
            break

wanted = {x["source1_entity_id"] for x in chosen}
for x in chosen:
    wanted.update(x["matched_entity_ids"].split(",") if x["matched_entity_ids"] else [])
records = {}
for source in (1, 2, 3):
    with (root / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in wanted:
                records[row["entity_id"]] = row

for x in chosen:
    ids = [x["source1_entity_id"]] + (x["matched_entity_ids"].split(",") if x["matched_entity_ids"] else [])
    print(json.dumps({"kind": x["kind"], "records": [records[i] for i in ids]}, ensure_ascii=False))
