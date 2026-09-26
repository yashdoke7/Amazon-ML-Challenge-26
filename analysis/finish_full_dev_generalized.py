"""Postprocess and evaluate the 22,133-query generalized model pass."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from number_veto_full_development import score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DB = ROOT / ".duckdb" / "train_index.duckdb"
PACKAGE = ROOT / "code" / "business_entity_resolution"
TMP = ROOT / "tmp" / "development_full"
MODEL = ROOT / "analysis" / "generalized_model.joblib"


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def predictions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row["matched_entity_ids"].split(","))
                if row["matched_entity_ids"] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    raw = TMP / "generalized_results.tsv"
    capped = TMP / "generalized_capped.tsv"
    final = TMP / "generalized_exclusive.tsv"
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", MODEL, "--input", raw, "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
        "--model", MODEL, "--input", capped, "--output", final)
    reference = predictions(TMP / "large_number_country_exclusive.tsv")
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            query = row["source1_entity_id"]
            if query in reference:
                truth[query] = set(row["matched_entity_ids"].split(",")) \
                    if row["matched_entity_ids"] else set()
    countries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in reference:
                countries[row["entity_id"]] = row["country"]
    assert set(reference) == set(truth) == set(countries)
    result = {"queries": len(truth),
              "old_final": score(reference, truth, countries),
              "new_raw": score(predictions(raw), truth, countries),
              "new_capped": score(predictions(capped), truth, countries),
              "new_final": score(predictions(final), truth, countries)}
    for value in result.values():
        if isinstance(value, dict) and "country_macro_f05" in value:
            c = value["country_macro_f05"]
            value["known_test_mix_f05"] = round(
                (0.3827 * c["US"] + 0.4675 * c["India"]) / (0.3827 + 0.4675), 6)
    destination = ROOT / "analysis" / "full_dev_generalized_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
