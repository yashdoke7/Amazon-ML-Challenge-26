"""Cost and optimistic miss coverage of label-free anchor-query gates."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def read(path, field, quota=None):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")[:quota])
                if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    base = read(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    address = read(ROOT / "analysis/india_address_tfidf_dev_candidates.tsv",
                   "candidate_entity_ids", 20)
    selected = read(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    truth = read(DATA / "train_ground_truth.tsv", "matched_entity_ids")
    counts = Counter()
    for q, pool in base.items():
        chosen = selected[q]
        missing = truth[q] - (pool | address.get(q, set()))
        size = str(min(len(chosen), 5)) + ("+" if len(chosen) >= 5 else "")
        source_mask = "+".join(sorted({t.split("-", 1)[0] for t in chosen})) or "none"
        for key in (f"size={size}", f"sources={source_mask}",
                    "size_le_1" if len(chosen) <= 1 else "size_gt_1",
                    "size_le_2" if len(chosen) <= 2 else "size_gt_2",
                    "size_le_3" if len(chosen) <= 3 else "size_gt_3"):
            counts[f"{key}|queries"] += 1
            counts[f"{key}|anchors"] += len(chosen)
            counts[f"{key}|missing_true_links"] += len(missing)
            counts[f"{key}|missing_with_true_anchor"] += sum(
                bool(truth[q] & chosen) for _ in missing)
    result = dict(sorted(counts.items()))
    dest = ROOT / "analysis/anchor_gate_headroom_results.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
