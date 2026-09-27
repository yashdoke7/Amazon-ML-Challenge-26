"""Probe whether row positions or numeric ID parts encode true entity groups.

Uses a deterministic 1-in-220 sample of train ground truth; aggregate output only.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def main():
    groups = {}
    wanted = set()
    for i, row in enumerate(rows(TRAIN / "train_ground_truth.tsv")):
        if i % 220:
            continue
        ids = [x for x in (row["matched_entity_ids"] or "").split(",") if x]
        groups[row["source1_entity_id"]] = ids
        wanted.add(row["source1_entity_id"])
        wanted.update(ids)
    positions = {}
    counts = {}
    for source in (1, 2, 3):
        count = 0
        for count, row in enumerate(rows(TRAIN / f"train_source{source}.tsv"), start=1):
            if row["entity_id"] in wanted:
                positions[row["entity_id"]] = count
        counts[source] = count
    values = defaultdict(list)
    for s1, target_ids in groups.items():
        if s1 not in positions:
            continue
        for source in (2, 3):
            matched = [x for x in target_ids if x.startswith(f"S{source}-") and x in positions]
            for i, x in enumerate(matched):
                values[f"s1_to_s{source}_row_gap"].append(
                    abs(positions[s1] / counts[1] - positions[x] / counts[source])
                )
                values[f"s1_to_s{source}_numeric_gap"].append(
                    abs(int(s1.split("-")[1]) - int(x.split("-")[1])) / 1e9
                )
                for y in matched[i + 1 :]:
                    values[f"within_s{source}_row_gap"].append(
                        abs(positions[x] - positions[y]) / counts[source]
                    )
                    values[f"within_s{source}_numeric_gap"].append(
                        abs(int(x.split("-")[1]) - int(y.split("-")[1])) / 1e9
                    )
    result = {
        "sampled_groups": len(groups),
        "source_rows": counts,
        "gaps": {
            k: {"n": len(v), "median": float(np.median(v)), "p10": float(np.quantile(v, 0.1))}
            for k, v in values.items()
        },
    }
    path = ROOT / "analysis/fresh_order_audit_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
