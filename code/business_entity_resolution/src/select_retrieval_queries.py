"""Select low-coverage Source 1 rows for a bounded follow-up retrieval route."""

import argparse
import csv
import itertools
from pathlib import Path


def run(data_dir, matching_path, output_path, country, max_matches):
    if max_matches < 0:
        raise ValueError("max-matches cannot be negative")
    query_path = data_dir / f"{data_dir.name}_source1.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = eligible = 0
    with query_path.open(encoding="utf-8", newline="") as query_stream, \
         matching_path.open(encoding="utf-8", newline="") as match_stream, \
         output_path.open("w", encoding="utf-8", newline="") as destination:
        queries = csv.DictReader(query_stream, delimiter="\t")
        matches = csv.DictReader(match_stream, delimiter="\t")
        if matches.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected matching TSV header")
        for query, match in itertools.zip_longest(queries, matches):
            if query is None or match is None or query["entity_id"] != match[
                    "source1_entity_id"]:
                raise ValueError("Matching rows differ from Source 1 order")
            total += 1
            if query["country"] != country:
                continue
            values = match["matched_entity_ids"]
            count = values.count(",") + 1 if values else 0
            if count <= max_matches:
                destination.write(query["entity_id"] + "\n")
                eligible += 1
    print("COMPLETE", total, "eligible", eligible, "country", country,
          "max_matches", max_matches, "output", output_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--country", default="US")
    parser.add_argument("--max-matches", type=int, default=3)
    args = parser.parse_args()
    run(args.data_dir, args.matching, args.output, args.country, args.max_matches)
