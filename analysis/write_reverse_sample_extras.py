"""Convert label-blind reverse retrieval pairs into sparse evaluator input."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.0)
    parser.add_argument("--field", choices=("both", "business_name", "business_address"),
                        default="both")
    args = parser.parse_args()
    if not 0 <= args.min_cosine <= 1:
        parser.error("min-cosine must be in [0,1]")
    groups = defaultdict(set)
    retained = 0
    with args.pairs.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if args.field != "both" and row["field"] != args.field:
                continue
            if float(row["cosine"]) >= args.min_cosine:
                groups[row["source1_entity_id"]].add(row["target_id"])
                retained += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for q in sorted(groups):
            writer.writerow([q, ",".join(sorted(groups[q]))])
    print("raw pairs", retained, "unique pairs", sum(map(len, groups.values())),
          "queries", len(groups), "max_per_query", max(map(len, groups.values()), default=0),
          "output", args.output, flush=True)


if __name__ == "__main__":
    main()
