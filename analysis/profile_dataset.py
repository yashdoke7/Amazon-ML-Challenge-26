"""Streaming profile of the supplied 2026 challenge files (no external data)."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"


def profile_source(path):
    countries = Counter()
    missing = Counter()
    examples = {}
    n = 0
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n += 1
            country = row["country"]
            countries[country] += 1
            for col in ("business_name", "business_address", "country"):
                if not row[col].strip():
                    missing[col] += 1
            if country not in examples:
                examples[country] = row
    return {"rows": n, "countries": countries, "missing": missing, "country_examples": examples}


def profile_truth(path):
    counts = Counter()
    source_mix = Counter()
    total_edges = 0
    n = 0
    duplicate_ids = 0
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n += 1
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            counts[len(ids)] += 1
            total_edges += len(ids)
            source_mix[(sum(x.startswith("S2-") for x in ids), sum(x.startswith("S3-") for x in ids))] += 1
            duplicate_ids += len(ids) != len(set(ids))
    return {"rows": n, "match_count_distribution": counts, "edges": total_edges,
            "source_count_combinations_top": source_mix.most_common(20), "rows_with_duplicate_matches": duplicate_ids}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            key = f"{split}_source{source}"
            result[key] = profile_source(ROOT / split / f"{key}.tsv")
            print(key, json.dumps(result[key], ensure_ascii=False), flush=True)
    result["train_ground_truth"] = profile_truth(ROOT / "train" / "train_ground_truth.tsv")
    print("train_ground_truth", json.dumps(result["train_ground_truth"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
