"""Evaluate a new blank-address specialist on the fixed final candidate pool."""

import argparse
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

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DB = ROOT / ".duckdb" / "train_index.duckdb"
TMP = ROOT / "tmp" / "development_full"
PACKAGE = ROOT / "code" / "business_entity_resolution"


def read_ids(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")) if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float,
                        help="Fixed development-selected threshold for frozen validation")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.split == "development":
        candidate_path = TMP / "tfidf_top20_candidates.tsv"
        baseline_raw = TMP / "tfidf_top20_raw.tsv"
        baseline_final = TMP / "tfidf_top20_final.tsv"
    else:
        candidate_path = TMP / "validation_tfidf_merged_candidates.tsv"
        baseline_raw = TMP / "validation_tfidf_merged_raw.tsv"
        baseline_final = TMP / "validation_tfidf_merged_final.tsv"
    candidate = read_ids(candidate_path, "candidate_entity_ids")
    selected = read_ids(baseline_raw, "matched_entity_ids")
    assert set(candidate) == set(selected)
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in candidate:
                queries[row["entity_id"]] = row
    assert set(queries) == set(candidate)
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in candidate:
                truth[row["source1_entity_id"]] = (
                    set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"]
                    else set())
    assert set(truth) == set(candidate)
    countries = {q: row["country"] for q, row in queries.items()}

    pairs = [(q, target) for q, ids in candidate.items() for target in ids]
    con = duckdb.connect(str(DB), read_only=True)
    con.register("candidate_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id"]))
    frame = con.execute("""SELECT p.s1_id,p.target_id,t.business_name t_name,
        coalesce(t.business_address,'') t_address,t.source target_source,
        f.df target_core_df FROM candidate_pairs p JOIN target t USING(target_id)
        JOIN blank_target_frequency f USING(target_id)
        WHERE coalesce(t.business_address,'')=''""").df()
    con.close()
    assert frame.target_core_df.notna().all()
    features = np.empty((len(frame), 36), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        q = queries[row.s1_id]
        features[i, :35] = generalized_pair_features(
            q["business_name"], row.t_name, q["business_address"], "", row.target_source)
        features[i, 35] = np.log1p(row.target_core_df)
    model = joblib.load(args.model)
    probabilities = model.predict_proba(features)[:, 1]
    blanks = defaultdict(set)
    for q, target in frame[["s1_id", "target_id"]].itertuples(index=False, name=None):
        blanks[q].add(target)
    thresholds = ([args.threshold] if args.threshold is not None else
                  [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    results = {"baseline": score(read_ids(baseline_final, "matched_entity_ids"),
                                 truth, countries),
               "queries": len(candidate), "blank_candidate_pairs": len(frame),
               "thresholds": {}}
    for threshold in thresholds:
        additions = defaultdict(set)
        for (q, target), p in zip(frame[["s1_id", "target_id"]].itertuples(
                index=False, name=None), probabilities):
            if p >= threshold:
                additions[q].add(target)
        revised = {q: (selected[q] - blanks[q]) | additions[q] for q in selected}
        stem = f"{args.tag}_{args.split}_{str(threshold).replace('.', 'p')}"
        raw, capped, final = [TMP / f"{stem}_{stage}.tsv" for stage in
                              ("raw", "capped", "final")]
        with raw.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for q, ids in revised.items():
                writer.writerow([q, ",".join(sorted(ids))])
        run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
            "--model", PACKAGE / "generalized_model.joblib", "--input", raw,
            "--output", capped, "--cap", "11")
        run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
            "--model", PACKAGE / "generalized_model.joblib", "--input", capped,
            "--output", final)
        measured = score(read_ids(final, "matched_entity_ids"), truth, countries)
        results["thresholds"][str(threshold)] = {
            "final": measured,
            "blank_selected": sum(map(len, additions.values())),
            "blank_tp": sum(len(additions[q] & truth[q]) for q in additions),
            "blank_fp": sum(len(additions[q] - truth[q]) for q in additions)}
        print("threshold", threshold, "macro", measured["macro_f05"], flush=True)
    output = ROOT / "analysis" / f"{args.tag}_{args.split}_results.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
