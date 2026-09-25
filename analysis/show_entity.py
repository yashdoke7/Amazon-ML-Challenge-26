"""Print one or more S1 records and their labeled target records."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("ids", nargs="+")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
labels = {}
wanted = set(args.ids)
with (root / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["source1_entity_id"] in wanted:
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            labels[row["source1_entity_id"]] = ids
            wanted.update(ids)
records = {}
for source in (1, 2, 3):
    with (root / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in wanted:
                records[row["entity_id"]] = row
for s1 in args.ids:
    print(json.dumps({"s1": records.get(s1), "matches": [records[i] for i in labels.get(s1, [])]}, ensure_ascii=False))
