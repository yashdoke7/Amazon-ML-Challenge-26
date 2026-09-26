"""Compare aggregate output tails by country without exposing test records."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_predictions(path):
    predictions = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            predictions[row["source1_entity_id"]] = tuple(
                item for item in row["matched_entity_ids"].split(",") if item
            )
    return predictions


def summary(predictions, countries):
    sizes = defaultdict(list)
    for entity_id, targets in predictions.items():
        sizes[countries[entity_id]].append(len(targets))
    result = {}
    for country, values in sizes.items():
        ordered = sorted(values)
        n = len(ordered)
        result[country] = {
            "queries": n,
            "pairs": sum(ordered),
            "empty": ordered.count(0),
            "over_11": sum(value > 11 for value in ordered),
            "over_50": sum(value > 50 for value in ordered),
            "p95": ordered[int(0.95 * (n - 1))],
            "p99": ordered[int(0.99 * (n - 1))],
            "max": ordered[-1],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--alternative", required=True)
    args = parser.parse_args()
    reference = read_predictions(args.reference)
    alternative = read_predictions(args.alternative)
    if reference.keys() != alternative.keys():
        raise ValueError("Prediction files contain different query IDs")
    countries = {}
    with Path(args.source1).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = row["entity_id"]
            if entity_id in reference:
                countries[entity_id] = row["country"]
            if len(countries) == len(reference):
                break
    if len(countries) != len(reference):
        raise ValueError("Some predicted query IDs are missing from Source 1")
    changed = Counter()
    added = Counter()
    removed = Counter()
    for entity_id in reference:
        country = countries[entity_id]
        before, after = set(reference[entity_id]), set(alternative[entity_id])
        changed[country] += before != after
        added[country] += len(after - before)
        removed[country] += len(before - after)
    print(json.dumps({
        "reference": summary(reference, countries),
        "alternative": summary(alternative, countries),
        "changed_queries": changed,
        "added_pairs": added,
        "removed_pairs": removed,
    }, indent=2))


if __name__ == "__main__":
    main()
