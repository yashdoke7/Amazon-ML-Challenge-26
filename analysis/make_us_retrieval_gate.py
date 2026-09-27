"""Select low-coverage US queries for the measured address rescue route."""

import csv
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
MATCH = ROOT / "output" / "generalized_compact_results.tsv"
DEST = ROOT / "tmp" / "us_address_gate_ids.txt"


def main():
    all_queries = us_queries = gated = 0
    with (DATA / "test_source1.tsv").open(encoding="utf-8", newline="") as qstream, \
         MATCH.open(encoding="utf-8", newline="") as mstream, \
         DEST.open("w", encoding="utf-8", newline="") as output:
        queries = csv.DictReader(qstream, delimiter="\t")
        matches = csv.DictReader(mstream, delimiter="\t")
        for query, match in itertools.zip_longest(queries, matches):
            if query is None or match is None or query["entity_id"] != match["source1_entity_id"]:
                raise ValueError("Source 1 and verified result rows differ")
            all_queries += 1
            if query["country"] == "US":
                us_queries += 1
                ids = match["matched_entity_ids"]
                count = ids.count(",") + 1 if ids else 0
                if count <= 3:
                    output.write(query["entity_id"] + "\n")
                    gated += 1
    assert all_queries == 1_732_544 and us_queries == 663_106
    print("all", all_queries, "US", us_queries, "US selected count<=3", gated,
          "output", DEST, flush=True)


if __name__ == "__main__":
    main()
