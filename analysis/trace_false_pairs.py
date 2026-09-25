"""Trace hard collision targets to their labeled S1 owner, if any."""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
x = json.loads((Path(__file__).resolve().parent / "collision_eda_results.json").read_text(encoding="utf-8"))
pairs = [(route, p) for route in ("exact_name", "exact_address") for p in x[route]["hardest_false_examples"][:12]]
targets = {p["target_id"] for _, p in pairs}
owner = {}
with (root / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        for target in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []:
            if target in targets:
                owner[target] = row["source1_entity_id"]
owners = set(owner.values())
records = {}
with (root / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in owners:
            records[row["entity_id"]] = row
for route, p in pairs:
    oid = owner.get(p["target_id"])
    print(json.dumps({"route": route, "query": [p["s1_id"], p["s1_name"], p["s1_address"]],
                      "candidate": [p["target_id"], p["target_name"], p["target_address"]],
                      "candidate_owner": records.get(oid) if oid else None,
                      "other_field_similarity": p["other_field_similarity"]}, ensure_ascii=False))
