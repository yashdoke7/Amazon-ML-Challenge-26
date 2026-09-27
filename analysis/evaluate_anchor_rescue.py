"""Score a one-hop predicted-anchor rescue on a fixed labeled query sample.

Development chooses the rule. Validation must use a previously chosen rule;
this tool reports a grid for diagnosis, not permission to tune on validation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from validation import f05_for_query  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"
MODEL = ROOT / "code/business_entity_resolution/generalized_model.joblib"
SPECIALIST = ROOT / "code/business_entity_resolution/blank_frequency_model.joblib"


def read(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")) if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--quota", type=int)
    parser.add_argument("--anchor-threshold", type=float)
    parser.add_argument("--query-threshold", type=float)
    args = parser.parse_args()
    if args.split == "validation" and (args.quota is None or
                                       args.anchor_threshold is None or
                                       args.query_threshold is None):
        parser.error("Validation requires a frozen quota and both thresholds")
    qids = set(args.query_ids.read_text(encoding="utf-8").splitlines())
    baseline_path = (DEV / "tfidf_top20_final.tsv" if args.split == "development"
                     else DEV / "validation_tfidf_merged_final.tsv")
    baseline = read(baseline_path, "matched_entity_ids")
    truth = read(DATA / "train_ground_truth.tsv", "matched_entity_ids")
    assert qids <= baseline.keys() and qids <= truth.keys()
    with args.pairs.open(encoding="utf-8", newline="") as stream:
        pairs = pd.DataFrame(csv.DictReader(stream, delimiter="\t"))
    if pairs.empty:
        raise ValueError("No anchor candidates")
    if not set(pairs.source1_entity_id) <= qids:
        raise ValueError("Candidate queries outside sampled query set")
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in qids:
                queries[row["entity_id"]] = row
    assert set(queries) == qids
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.register("anchor_pairs", pairs[["source1_entity_id", "target_id"]])
    frame = con.execute("""SELECT p.source1_entity_id,p.target_id,
        t.business_name,t.business_address,t.source,t.country,
        f.df target_core_frequency FROM anchor_pairs p JOIN target t USING(target_id)
        LEFT JOIN blank_target_frequency f USING(target_id)""").df()
    con.close()
    if len(frame) != len(pairs):
        raise ValueError("Missing target metadata")
    frame = frame.merge(pairs, on=["source1_entity_id", "target_id"],
                        validate="one_to_one")
    model = joblib.load(MODEL)
    specialist = joblib.load(SPECIALIST)
    X = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        q = queries[row.source1_entity_id]
        X[i] = generalized_pair_features(q["business_name"], row.business_name,
                                         q["business_address"], row.business_address or "",
                                         row.source)
    probabilities = model.predict_proba(X)[:, 1]
    blank = frame.business_address.fillna("").eq("").to_numpy()
    if blank.any():
        frequencies = frame.loc[blank, "target_core_frequency"]
        if frequencies.isna().any():
            raise ValueError("Blank target missing core-name frequency")
        extended = np.column_stack((X[blank], np.log1p(
            frequencies.to_numpy(dtype=np.float32))))
        probabilities[blank] = specialist.predict_proba(extended)[:, 1]
    frame["query_probability"] = probabilities
    frame["anchor_probability"] = frame.anchor_probability.astype(float)
    frame["anchor_rank"] = frame.anchor_rank.astype(int)
    frame["is_true"] = [t in truth[q] for q, t in frame[[
        "source1_entity_id", "target_id"]].itertuples(index=False, name=None)]
    countries = {q: row["country"] for q, row in queries.items()}
    baseline_score = sum(f05_for_query(truth[q], baseline[q]) for q in qids) / len(qids)
    rows = []
    quotas = (args.quota,) if args.split == "validation" else (1, 3, 5)
    anchor_thresholds = ((args.anchor_threshold,) if args.split == "validation"
                         else (0.5, 0.7, 0.9, 0.95, 0.99))
    query_thresholds = ((args.query_threshold,) if args.split == "validation"
                        else (0.0, 0.1, 0.3, 0.5, 0.7))
    for quota in quotas:
        for anchor_threshold in anchor_thresholds:
            for query_threshold in query_thresholds:
                eligible = frame.loc[(frame.anchor_rank <= quota) &
                                     (frame.anchor_probability >= anchor_threshold) &
                                     (frame.query_probability >= query_threshold)]
                additions = {q: set(group.target_id) for q, group in eligible.groupby(
                    "source1_entity_id")}
                revised_score = sum(f05_for_query(truth[q], baseline[q] |
                                    additions.get(q, set())) for q in qids) / len(qids)
                rows.append({"quota": quota, "anchor_threshold": anchor_threshold,
                             "query_threshold": query_threshold,
                             "macro_f05": revised_score,
                             "gain": revised_score - baseline_score,
                             "new_pairs": len(eligible),
                             "true_pairs": int(eligible.is_true.sum()),
                             "false_pairs": int((~eligible.is_true).sum())})
    ranked = sorted(rows, key=lambda item: (-item["macro_f05"], item["new_pairs"]))
    result = {"split": args.split, "queries": len(qids), "countries": {
        country: sum(countries[q] == country for q in qids) for country in set(countries.values())},
        "baseline_macro_f05": baseline_score, "candidate_pairs": len(frame),
        "all_candidate_true_pairs": int(frame.is_true.sum()),
        "top_rules": ranked[:20],
        "all_rules": rows,
        "note": "Before group cap and exclusive ownership; select rule on development only."}
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "all_rules"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
