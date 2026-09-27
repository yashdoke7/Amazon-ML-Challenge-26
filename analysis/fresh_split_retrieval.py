"""Separate combined diagnostic retrieval output by fixed query-ID sets."""

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ids = set(args.ids.read_text(encoding="utf-8").splitlines())
    count = 0
    with args.input.open(encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as dest:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if row["source1_entity_id"] in ids:
                writer.writerow(row)
                count += 1
    print("COMPLETE", count)


if __name__ == "__main__":
    main()
