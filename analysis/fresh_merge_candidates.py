"""Build a diagnostic candidate union for comparing a new route fairly."""

import argparse
import csv
from pathlib import Path


def read_extra(path, quota):
    out = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            out[row["source1_entity_id"]] = set(filter(None, (row["candidate_entity_ids"] or "").split(",")[:quota]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="+", required=True)
    p.add_argument("--quota", type=int, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(args.extra) != len(args.quota):
        p.error("one quota required per extra file")
    extra = [read_extra(path, quota) for path, quota in zip(args.extra, args.quota)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = 0
    with args.base.open(encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.writer(dest, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in csv.DictReader(source, delimiter="\t"):
            q = row["source1_entity_id"]
            ids = set(filter(None, (row["candidate_entity_ids"] or "").split(",")))
            for route in extra:
                ids.update(route.get(q, ()))
            writer.writerow([q, ",".join(sorted(ids))])
            pairs += len(ids)
    print("COMPLETE", pairs)


if __name__ == "__main__":
    main()
