"""Build development or frozen-validation query IDs for a fixed prediction gate."""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--country", default="US")
    parser.add_argument("--max-matches", type=int, default=3)
    args = parser.parse_args()
    with args.matching.open(encoding="utf-8", newline="") as stream:
        counts = {row["source1_entity_id"]: (row["matched_entity_ids"].count(",") + 1
                  if row["matched_entity_ids"] else 0)
                  for row in csv.DictReader(stream, delimiter="\t")}
    eligible = 0
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as destination:
        for row in csv.DictReader(source, delimiter="\t"):
            q = row["entity_id"]
            if q in counts and row["country"] == args.country and counts[q] <= args.max_matches:
                destination.write(q + "\n")
                eligible += 1
    print("COMPLETE", eligible, flush=True)


if __name__ == "__main__":
    main()
