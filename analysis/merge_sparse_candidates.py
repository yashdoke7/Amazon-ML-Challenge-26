"""Union a sparse retrieval TSV into a complete candidate TSV for labeled tests."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.extra.open(encoding="utf-8", newline="") as stream:
        extra = {r["source1_entity_id"]: set(r["candidate_entity_ids"].split(","))
                 if r["candidate_entity_ids"] else set()
                 for r in csv.DictReader(stream, delimiter="\t")}
    seen = set()
    added = 0
    with args.base.open(encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in reader:
            q = row["source1_entity_id"]
            seen.add(q)
            base = set(row["candidate_entity_ids"].split(",")) if row[
                "candidate_entity_ids"] else set()
            union = base | extra.get(q, set())
            added += len(union) - len(base)
            writer.writerow([q, ",".join(sorted(union))])
    if set(extra) - seen:
        raise ValueError("Extra query outside base candidate set")
    print("COMPLETE queries", len(seen), "added_pairs", added, flush=True)


if __name__ == "__main__":
    main()
