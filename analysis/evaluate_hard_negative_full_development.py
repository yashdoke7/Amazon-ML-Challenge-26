"""Score the new model on every fixed development query and the final top-40 pool."""

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
MODEL = ROOT / "analysis" / "hard_negative_generalized_model.joblib"


def run(*args):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def predictions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row["matched_entity_ids"].split(","))
                if row["matched_entity_ids"] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    evaluation = json.loads((ROOT / "analysis" / "hard_negative_generalized_evaluation_results.json").read_text())
    thresholds = evaluation["thresholds_selected_on_development_only"]
    reference = predictions(TMP / "generalized_exclusive.tsv")
    raw = TMP / "hard_negative_top40_raw.tsv"
    capped = TMP / "hard_negative_top40_capped.tsv"
    final = TMP / "hard_negative_top40_final.tsv"
    run(PACKAGE / "src" / "rescore_candidates.py", "--data-dir", DATA, "--db", DB,
        "--model", MODEL, "--candidate", TMP / "generalized_top40_candidates.tsv",
        "--output", raw, "--threshold", thresholds["other"],
        "--country-threshold", f"India:{thresholds['India']}",
        "--query-ids", TMP / "query_ids.txt", "--workers", "4")
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", MODEL, "--input", raw, "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
        "--model", MODEL, "--input", capped, "--output", final)
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
    result = {"queries": len(truth), "thresholds": thresholds,
              "reference_current35_final": score(reference, truth, countries),
              "new_raw": score(predictions(raw), truth, countries),
              "new_capped": score(predictions(capped), truth, countries),
              "new_final": score(predictions(final), truth, countries)}
    destination = ROOT / "analysis" / "hard_negative_full_development_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
