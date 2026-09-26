"""Unlabeled first-20k test transfer check for the generalized matcher.

This measures output shape and country drift, never accuracy. It leaves the
complete saved candidate and leaderboard TSVs untouched.
"""

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
PACKAGE = ROOT / "code" / "business_entity_resolution"
TEMP = ROOT / "tmp" / "test_generalized_slice"
LIMIT = 20_000


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def read_groups(path, wanted):
    groups = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            query = row["source1_entity_id"]
            if query in wanted:
                groups[query] = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
    assert set(groups) == wanted
    return groups


def summarize(groups, countries):
    sizes = defaultdict(list)
    for query, targets in groups.items():
        sizes[countries[query]].append(len(targets))
    return {country: {"queries": len(values), "links": sum(values),
                      "empty": values.count(0), "above_11": sum(n > 11 for n in values),
                      "above_50": sum(n > 50 for n in values),
                      "maximum": max(values)}
            for country, values in sizes.items()}


def main():
    TEMP.mkdir(exist_ok=True)
    candidate = TEMP / "candidate_pairs.tsv"
    query_ids = TEMP / "query_ids.txt"
    raw = TEMP / "generalized_raw.tsv"
    capped = TEMP / "generalized_capped.tsv"
    final = TEMP / "generalized_final.tsv"
    countries = {}
    with (DATA / "test_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            if i >= LIMIT:
                break
            countries[row["entity_id"]] = row["country"]
    query_ids.write_text("\n".join(countries) + "\n", encoding="utf-8")
    with (ROOT / "output" / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as source, \
         candidate.open("w", encoding="utf-8", newline="") as dest:
        reader = csv.reader(source, delimiter="\t")
        writer = csv.writer(dest, delimiter="\t", lineterminator="\n")
        writer.writerow(next(reader))
        for i, row in enumerate(reader):
            if i >= LIMIT:
                break
            writer.writerow(row)
    run(PACKAGE / "src" / "rescore_candidates.py", "--data-dir", DATA,
        "--db", ROOT / ".duckdb" / "test_index.duckdb",
        "--model", ROOT / "analysis" / "generalized_model.joblib",
        "--candidate", candidate, "--output", raw, "--threshold", "0.8",
        "--query-ids", query_ids, "--workers", "4")
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA,
        "--db", ROOT / ".duckdb" / "test_index.duckdb",
        "--model", ROOT / "analysis" / "generalized_model.joblib",
        "--input", raw, "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA,
        "--db", ROOT / ".duckdb" / "test_index.duckdb",
        "--model", ROOT / "analysis" / "generalized_model.joblib",
        "--input", capped, "--output", final)
    wanted = set(countries)
    output = {
        "queries": LIMIT,
        "old_baseline_final": summarize(
            read_groups(ROOT / "output" / "matching_results.tsv", wanted), countries),
        "generalized_raw": summarize(read_groups(raw, wanted), countries),
        "generalized_final": summarize(read_groups(final, wanted), countries),
    }
    (ROOT / "analysis" / "generalized_test_slice_results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
