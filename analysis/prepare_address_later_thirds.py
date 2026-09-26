"""Write ID lists for the second and third exact India address-search shards."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" /
          "dataset" / "test" / "test_source1.tsv")
EXPECTED = 809_986
FIRST = EXPECTED // 3
SECOND = FIRST * 2


def main():
    count = 0
    paths = [ROOT / "tmp" / f"india_address_shard_{i}_ids.txt" for i in (2, 3)]
    with SOURCE.open(encoding="utf-8", newline="") as source, \
         paths[0].open("w", encoding="utf-8", newline="") as middle, \
         paths[1].open("w", encoding="utf-8", newline="") as last:
        for row in csv.DictReader(source, delimiter="\t"):
            if row["country"] != "India":
                continue
            if FIRST <= count < SECOND:
                middle.write(row["entity_id"] + "\n")
            elif count >= SECOND:
                last.write(row["entity_id"] + "\n")
            count += 1
    if count != EXPECTED:
        raise ValueError(f"Expected {EXPECTED} India queries, found {count}")
    print("India queries", count, "shard sizes", FIRST, SECOND - FIRST,
          EXPECTED - SECOND, "ID files", *paths)


if __name__ == "__main__":
    main()
