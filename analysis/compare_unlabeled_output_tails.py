"""Stream aggregate output-tail differences by country without retaining records."""

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def quantile(counts, index):
    seen = 0
    for size in sorted(counts):
        seen += counts[size]
        if seen > index:
            return size
    raise ValueError("Empty group distribution")


def summary(stats):
    result = {}
    for country, sizes in stats.items():
        n = sum(sizes.values())
        result[country] = {
            "queries": n,
            "pairs": sum(size * count for size, count in sizes.items()),
            "empty": sizes[0],
            "over_11": sum(count for size, count in sizes.items() if size > 11),
            "over_50": sum(count for size, count in sizes.items() if size > 50),
            "p95": quantile(sizes, int(0.95 * (n - 1))),
            "p99": quantile(sizes, int(0.99 * (n - 1))),
            "max": max(sizes),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--alternative", required=True)
    args = parser.parse_args()
    counts = {"reference": defaultdict(Counter), "alternative": defaultdict(Counter)}
    changed, added, removed = Counter(), Counter(), Counter()
    with Path(args.source1).open(newline="", encoding="utf-8") as source_stream, \
         Path(args.reference).open(newline="", encoding="utf-8") as reference_stream, \
         Path(args.alternative).open(newline="", encoding="utf-8") as alternative_stream:
        source = csv.DictReader(source_stream, delimiter="\t")
        reference = csv.DictReader(reference_stream, delimiter="\t")
        alternative = csv.DictReader(alternative_stream, delimiter="\t")
        for before_row, after_row in itertools.zip_longest(reference, alternative):
            if before_row is None or after_row is None:
                raise ValueError("Prediction files contain different row counts")
            query = next(source, None)
            if query is None:
                raise ValueError("More predictions than Source 1 rows")
            entity_id = query["entity_id"]
            if (before_row["source1_entity_id"] != entity_id or
                    after_row["source1_entity_id"] != entity_id):
                raise ValueError(f"Prediction order or query ID mismatch: {entity_id}")
            country = query["country"]
            before = set(filter(None, before_row["matched_entity_ids"].split(",")))
            after = set(filter(None, after_row["matched_entity_ids"].split(",")))
            counts["reference"][country][len(before)] += 1
            counts["alternative"][country][len(after)] += 1
            changed[country] += before != after
            added[country] += len(after - before)
            removed[country] += len(before - after)
    print(json.dumps({
        "reference": summary(counts["reference"]),
        "alternative": summary(counts["alternative"]),
        "changed_queries": changed,
        "added_pairs": added,
        "removed_pairs": removed,
    }, indent=2))


if __name__ == "__main__":
    main()
