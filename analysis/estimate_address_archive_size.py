"""Estimate ZIP candidate growth from the measured development India quota."""

import csv
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp" / "development_full" / "generalized_top40_candidates.tsv"
EXTRA = ROOT / "analysis" / "india_address_tfidf_dev_candidates.tsv"


def compressed_bytes(data):
    return len(zlib.compress(data, level=9))


def main():
    with EXTRA.open(encoding="utf-8", newline="") as stream:
        extra = {row["source1_entity_id"]: row["candidate_entity_ids"].split(",")
                 if row["candidate_entity_ids"] else []
                 for row in csv.DictReader(stream, delimiter="\t")}
    old = BASE.read_bytes()
    for quota in (5, 10, 20, 50):
        lines = ["source1_entity_id\tcandidate_entity_ids\n"]
        with BASE.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                q = row["source1_entity_id"]
                base = set(row["candidate_entity_ids"].split(",")) if row[
                    "candidate_entity_ids"] else set()
                ids = base | set(extra.get(q, [])[:quota])
                lines.append(q + "\t" + ",".join(sorted(ids)) + "\n")
        new = "".join(lines).encode("utf-8")
        print(quota, "development_zlib_growth_bytes",
              compressed_bytes(new) - compressed_bytes(old),
              "development_raw_growth_bytes", len(new) - len(old))


if __name__ == "__main__":
    main()
