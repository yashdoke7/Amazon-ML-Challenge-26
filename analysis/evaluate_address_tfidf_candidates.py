"""End-to-end development score for a small TF-IDF address retrieval quota."""

import csv
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from generalized_features import GENERALIZED_FEATURE_NAMES  # noqa: E402
from rescore_candidates import featurize_rows  # noqa: E402
from number_veto_full_development import score  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DB = ROOT / ".duckdb" / "train_index.duckdb"
TMP = ROOT / "tmp" / "development_full"
PACKAGE = ROOT / "code" / "business_entity_resolution"
EXTRA = ROOT / "analysis" / "india_address_tfidf_dev_candidates.tsv"
QUOTAS = (1, 5, 10, 20, 50)


def read_ids(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")) if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prefix", default="blank_frequency_top40")
    parser.add_argument("--extra", type=Path, default=EXTRA)
    parser.add_argument("--base-candidate", type=Path)
    parser.add_argument("--country", default="India")
    parser.add_argument("--general-threshold", type=float, default=0.65)
    parser.add_argument("--tag", default="")
    parser.add_argument("--quotas", type=int, nargs="+", default=list(QUOTAS))
    parser.add_argument("--result", type=Path,
                        default=ROOT / "analysis" / "india_address_tfidf_end_to_end_results.json")
    args = parser.parse_args()
    if not args.quotas or min(args.quotas) < 1:
        parser.error("Provide positive quotas")
    started = time.perf_counter()
    candidate_path = args.base_candidate or (TMP / "generalized_top40_candidates.tsv" if
                      args.base_prefix == "blank_frequency_top40" else
                      TMP / f"{args.base_prefix}_candidates.tsv")
    base_candidates = read_ids(candidate_path, "candidate_entity_ids")
    base_raw = read_ids(TMP / f"{args.base_prefix}_raw.tsv", "matched_entity_ids")
    base_final = read_ids(TMP / f"{args.base_prefix}_final.tsv", "matched_entity_ids")
    assert set(base_candidates) == set(base_raw) == set(base_final)
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in base_candidates:
                queries[row["entity_id"]] = row
    pairs = []
    with args.extra.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            assert q in queries and queries[q]["country"] == args.country
            ids = row["candidate_entity_ids"].split(",") if row["candidate_entity_ids"] else []
            for rank, target in enumerate(ids[:max(args.quotas)]):
                if target not in base_candidates[q]:
                    pairs.append((q, target, rank))
    con = duckdb.connect(str(DB), read_only=True)
    con.register("extra_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id", "extra_rank"]))
    frame = con.execute("""SELECT p.s1_id,p.target_id,p.extra_rank,t.business_name t_name,
        coalesce(t.business_address,'') t_address,t.source target_source,
        f.df target_core_df FROM extra_pairs p JOIN target t USING(target_id)
        LEFT JOIN blank_target_frequency f USING(target_id)""").df().fillna({"t_name": "", "t_address": ""})
    con.close()
    assert len(frame) == len(pairs)
    assert frame.loc[frame.t_address == "", "target_core_df"].notna().all()
    print("new pairs", len(frame), "seconds", round(time.perf_counter()-started, 1), flush=True)
    feature_rows = [(queries[row.s1_id]["business_name"], row.t_name,
                     queries[row.s1_id]["business_address"], row.t_address,
                     row.target_source) for row in frame.itertuples(index=False)]
    chunks = (feature_rows[i:i+50_000] for i in range(0, len(feature_rows), 50_000))
    with ProcessPoolExecutor(max_workers=4) as pool:
        X = np.concatenate(list(pool.map(
            partial(featurize_rows, model_feature_names=GENERALIZED_FEATURE_NAMES), chunks)))
    base = joblib.load(PACKAGE / "generalized_model.joblib")
    specialist = joblib.load(PACKAGE / "blank_frequency_model.joblib")
    probabilities = base.predict_proba(X)[:, 1]
    blank = frame.t_address.to_numpy() == ""
    if blank.any():
        special_X = np.column_stack((X[blank], np.log1p(
            frame.loc[blank, "target_core_df"].to_numpy(dtype=np.float32))))
        probabilities[blank] = specialist.predict_proba(special_X)[:, 1]
    frame["selected"] = probabilities >= np.where(blank, 0.8, args.general_threshold)
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in queries:
                truth[q] = set(row["matched_entity_ids"].split(",")) \
                    if row["matched_entity_ids"] else set()
    countries = {q: row["country"] for q, row in queries.items()}
    result = {"baseline_final": score(base_final, truth, countries),
              "extra_pairs_scored": len(frame), "seconds_features": round(time.perf_counter()-started, 1),
              "quotas": {}}
    for quota in args.quotas:
        eligible = frame.loc[frame.extra_rank < quota]
        additions = defaultdict(set)
        candidate_extra = defaultdict(set)
        for row in eligible.itertuples(index=False):
            candidate_extra[row.s1_id].add(row.target_id)
            if row.selected:
                additions[row.s1_id].add(row.target_id)
        revised = {q: base_raw[q] | additions[q] for q in base_raw}
        stem = args.base_prefix + (f"_{args.tag}" if args.tag else "")
        raw = TMP / f"{stem}_tfidf_top{quota}_raw.tsv"
        capped = TMP / f"{stem}_tfidf_top{quota}_capped.tsv"
        final = TMP / f"{stem}_tfidf_top{quota}_final.tsv"
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
        scored = score(read_ids(final, "matched_entity_ids"), truth, countries)
        selected_true = {q: len(additions[q] & truth[q]) for q in additions}
        selected_false = {q: len(additions[q] - truth[q]) for q in additions}
        result["quotas"][str(quota)] = {
            "new_candidates": sum(map(len, candidate_extra.values())),
            "new_true_candidates": sum(len(candidate_extra[q] & truth[q]) for q in candidate_extra),
            "selected_additions": sum(map(len, additions.values())),
            "selected_true_additions": sum(selected_true.values()),
            "selected_false_additions": sum(selected_false.values()),
            "selected_true_on_previously_empty_queries": sum(
                count for q, count in selected_true.items() if not base_raw[q]),
            "selected_true_on_previously_nonempty_queries": sum(
                count for q, count in selected_true.items() if base_raw[q]),
            "raw": score(revised, truth, countries), "final": scored}
        print("quota", quota, "new_candidates", result["quotas"][str(quota)]["new_candidates"],
              "macro", scored["macro_f05"], "seconds", round(time.perf_counter()-started, 1), flush=True)
    result["seconds_total"] = round(time.perf_counter()-started, 1)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
