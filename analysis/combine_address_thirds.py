"""Combine exact TF-IDF outputs for ordered India query thirds.

The first input may have extra rows after a controlled stop. The other inputs
must contain exactly their assigned India queries. Every output row is checked
against supplied Source 1 order before writing the final file.
"""

import argparse
import csv
from pathlib import Path


EXPECTED = 809_986
FIRST = EXPECTED // 3
SECOND = FIRST * 2
HEADER = ["source1_entity_id", "candidate_entity_ids"]


def run(source1, first, middle, last, output,
        expected=EXPECTED, first_cut=FIRST, second_cut=SECOND):
    if len({p.resolve() for p in (source1, first, middle, last, output)}) != 5:
        raise ValueError("Input and output paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source1.open(encoding="utf-8", newline="") as source_stream, \
         first.open(encoding="utf-8", newline="") as first_stream, \
         middle.open(encoding="utf-8", newline="") as middle_stream, \
         last.open(encoding="utf-8", newline="") as last_stream, \
         output.open("w", encoding="utf-8", newline="") as destination:
        queries = (row for row in csv.DictReader(source_stream, delimiter="\t")
                   if row["country"] == "India")
        inputs = [csv.DictReader(stream, delimiter="\t") for stream in
                  (first_stream, middle_stream, last_stream)]
        if any(rows.fieldnames != HEADER for rows in inputs):
            raise ValueError("Unexpected address candidate header")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        count = 0
        for query in queries:
            shard = 0 if count < first_cut else 1 if count < second_cut else 2
            row = next(inputs[shard], None)
            if row is None or row["source1_entity_id"] != query["entity_id"]:
                raise ValueError(f"Missing or out-of-order India query at {count}")
            ids = row["candidate_entity_ids"].split(",") if row[
                "candidate_entity_ids"] else []
            if len(ids) != len(set(ids)) or len(ids) > 20:
                raise ValueError(f"Invalid candidate list at {count}")
            writer.writerow([query["entity_id"], row["candidate_entity_ids"]])
            count += 1
        if count != expected or any(next(rows, None) is not None for rows in inputs[1:]):
            raise ValueError("Unexpected number of India queries")
    print("COMPLETE", count, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source1", "first", "middle", "last", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected", type=int, default=EXPECTED)
    parser.add_argument("--first-cut", type=int, default=FIRST)
    parser.add_argument("--second-cut", type=int, default=SECOND)
    args = parser.parse_args()
    run(args.source1, args.first, args.middle, args.last, args.output,
        args.expected, args.first_cut, args.second_cut)
