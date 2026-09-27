"""Apply a fixed existing-match-count gate to a sparse retrieval TSV."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-matches", type=int, required=True)
    args = parser.parse_args()
    with args.matching.open(encoding="utf-8", newline="") as stream:
        counts = {row["source1_entity_id"]: (row["matched_entity_ids"].count(",") + 1
                  if row["matched_entity_ids"] else 0)
                  for row in csv.DictReader(stream, delimiter="\t")}
    written = 0
    with args.extra.open(encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames,
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            q = row["source1_entity_id"]
            if q not in counts:
                raise ValueError("Retrieval query outside matching file")
            if counts[q] <= args.max_matches:
                writer.writerow(row)
                written += 1
    print("COMPLETE", written, flush=True)


if __name__ == "__main__":
    main()
