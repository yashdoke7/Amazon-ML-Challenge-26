"""Label-blind reciprocal-rank fusion of two completed candidate routes."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path):
    out = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            out[row["source1_entity_id"]] = [x for x in
                (row["candidate_entity_ids"] or "").split(",") if x]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rank-constant", type=float, default=10.0)
    args = p.parse_args()
    first = load(args.first)
    second = load(args.second)
    assert first.keys() == second.keys()
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for q, left in first.items():
            scores = defaultdict(float)
            for route in (left, second[q]):
                for rank, target in enumerate(route, start=1):
                    scores[target] += 1.0 / (args.rank_constant + rank)
            best = sorted(scores, key=lambda target: (-scores[target], target))[:args.top_k]
            writer.writerow([q, ",".join(best)])
    print("COMPLETE", len(first))


if __name__ == "__main__":
    main()
