"""Exact top-40 full-development test of the blank-address frequency model."""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from generalized_features import generalized_pair_features  # noqa: E402
from number_veto_full_development import score  # noqa: E402
from probe_blank_name_frequency import name_frequencies  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DB = ROOT / ".duckdb" / "train_index.duckdb"
TMP = ROOT / "tmp" / "development_full"
PACKAGE = ROOT / "code" / "business_entity_resolution"


def read_ids(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")) if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def run(*args):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    selected = read_ids(TMP / "hard_negative_top40_raw.tsv", "matched_entity_ids")
    candidate = read_ids(TMP / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    assert set(selected) == set(candidate)
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in selected:
                queries[row["entity_id"]] = row
    pairs = [(q, t) for q, ids in candidate.items() for t in ids]
    con = duckdb.connect(str(DB), read_only=True)
    con.register("candidate_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id"]))
    frame = con.execute("""SELECT p.s1_id,p.target_id,t.business_name t_name,
        coalesce(t.business_address,'') t_address,t.source target_source
        FROM candidate_pairs p JOIN target t USING(target_id)
        WHERE coalesce(t.business_address,'')=''""").df().fillna("")
    con.close()
    assert len(frame) > 1000
    frequency = name_frequencies((frame,))
    features = np.empty((len(frame), 36), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        q = queries[row.s1_id]
        features[i, :35] = generalized_pair_features(
            q["business_name"], row.t_name, q["business_address"], row.t_address,
            row.target_source)
        features[i, 35] = np.log1p(frequency[row.target_id])
    specialist = joblib.load(ROOT / "analysis" / "blank_frequency_specialist_model.joblib")
    probabilities = specialist.predict_proba(features)[:, 1]
    threshold = json.loads((ROOT / "analysis" / "blank_frequency_specialist_results.json").read_text())[
        "chosen_on_development"]
    blanks = defaultdict(set)
    accepted = defaultdict(set)
    for row, p in zip(frame.itertuples(index=False), probabilities):
        blanks[row.s1_id].add(row.target_id)
        if p >= threshold:
            accepted[row.s1_id].add(row.target_id)
    revised = {q: (selected[q] - blanks[q]) | accepted[q] for q in selected}
    raw = TMP / "blank_frequency_top40_raw.tsv"
    with raw.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for q, ids in revised.items():
            writer.writerow([q, ",".join(sorted(ids))])
    capped = TMP / "blank_frequency_top40_capped.tsv"
    final = TMP / "blank_frequency_top40_final.tsv"
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "generalized_model.joblib", "--input", raw,
        "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "generalized_model.joblib", "--input", capped,
        "--output", final)
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in selected:
                truth[q] = set(row["matched_entity_ids"].split(",")) \
                    if row["matched_entity_ids"] else set()
    countries = {q: row["country"] for q, row in queries.items()}
    result = {"queries": len(selected), "blank_candidate_pairs": len(frame),
              "blank_selected_old": sum(len(selected[q] & blanks[q]) for q in selected),
              "blank_selected_new": sum(map(len, accepted.values())),
              "current_final": score(read_ids(TMP / "hard_negative_top40_final.tsv", "matched_entity_ids"),
                                     truth, countries),
              "specialist_raw": score(revised, truth, countries),
              "specialist_final": score(read_ids(final, "matched_entity_ids"), truth, countries)}
    destination = ROOT / "analysis" / "blank_frequency_full_development_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
