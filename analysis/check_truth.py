"""Inspect label graph constraints without loading record text into memory."""
import csv
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
targets = {2: array("I"), 3: array("I")}
counts = Counter()
with path.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        counts["s1_rows"] += 1
        counts["singletons"] += not ids
        for target in ids:
            source = int(target[1])
            targets[source].append(int(target[3:]))
            counts[f"s{source}_edges"] += 1
for source, values in targets.items():
    a = np.frombuffer(values, dtype=np.uint32)
    a.sort()
    unique = 1 + int(np.count_nonzero(a[1:] != a[:-1]))
    print(f"S{source}: edges={len(a)}, unique target IDs={unique}, reused by multiple S1={len(a)-unique}")
print(dict(counts))
