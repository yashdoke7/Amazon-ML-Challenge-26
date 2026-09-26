"""Evaluate the saved first matcher on a larger frozen validation sample."""

import csv
import json
import os
import random
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from train_first_matcher import DATA, FEATURE_NAMES, ROOT, entity_split, f05_for_query, pair_features

COMBINED = os.environ.get("EVAL_COMBINED") == "1"
COMBINED_MODEL = os.environ.get("EVAL_COMBINED_MODEL") == "1"
PAIRS = ROOT / "analysis" / ("full_validation_combined_pairs.parquet" if COMBINED else "full_validation_pairs.parquet")
suffix = "combined_retrained" if COMBINED and COMBINED_MODEL else "combined" if COMBINED else "large"
suffix = os.environ.get("EVAL_TAG",suffix)
METRICS = ROOT / "analysis" / f"first_matcher_{suffix}_validation_results.json"
ERRORS = ROOT / "analysis" / f"first_matcher_{suffix}_errors_results.json"
MODEL_PATH = ROOT / "analysis" / ("first_matcher_combined_model.joblib" if COMBINED_MODEL else "first_matcher_model.joblib")
if os.environ.get("EVAL_MODEL_PATH"):
    MODEL_PATH = ROOT / "analysis" / os.environ["EVAL_MODEL_PATH"]
TRAINING_RESULTS = ROOT / "analysis" / ("first_matcher_combined_training_results.json" if COMBINED_MODEL else "first_matcher_results.json")
N = int(os.environ.get("VALIDATION_QUERY_COUNT", "2000"))
THRESHOLD = float(os.environ.get("EVAL_THRESHOLD",
    json.loads(TRAINING_RESULTS.read_text(encoding="utf-8"))["calibration_threshold"]))


def sampled_truth():
    rng = random.Random(20260925)
    sample = []
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8",newline="") as stream:
        eligible = 0
        for row in csv.DictReader(stream,delimiter="\t"):
            if entity_split(row["source1_entity_id"]) != "validation":
                continue
            item = (row["source1_entity_id"],set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set())
            if eligible < N:
                sample.append(item)
            else:
                j = rng.randrange(eligible+1)
                if j < N:
                    sample[j] = item
            eligible += 1
    return dict(sample)


def main():
    tic = time.perf_counter()
    truth = sampled_truth()
    df = pd.read_parquet(PAIRS)
    assert set(df.s1_id).issubset(truth)
    assert len(df) == len(df[["s1_id","target_id"]].drop_duplicates())
    assert np.array_equal(df.is_match.to_numpy(),np.array([t in truth[s] for s,t in zip(df.s1_id,df.target_id)]))
    features = np.empty((len(df),len(FEATURE_NAMES)),dtype=np.float32)
    for i,row in enumerate(df.itertuples(index=False)):
        features[i] = pair_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source)
        if i and i % 100_000 == 0:
            print("features",i,"seconds",round(time.perf_counter()-tic,1),flush=True)
    model = joblib.load(MODEL_PATH)
    assert list(model.feature_name_) == FEATURE_NAMES
    probabilities = model.predict_proba(features)[:,1]
    df["probability"] = probabilities
    predictions = defaultdict(set)
    available = defaultdict(set)
    for row in df.itertuples(index=False):
        if row.is_match:
            available[row.s1_id].add(row.target_id)
        if row.probability >= THRESHOLD:
            predictions[row.s1_id].add(row.target_id)
    ids = list(truth)
    scores = [f05_for_query(truth[s],predictions[s]) for s in ids]
    oracle = [f05_for_query(truth[s],available[s]) for s in ids]
    tp = sum(len(predictions[s]&truth[s]) for s in ids)
    fp = sum(len(predictions[s]-truth[s]) for s in ids)
    fn = sum(len(truth[s]-predictions[s]) for s in ids)
    singleton_ids = [s for s in ids if not truth[s]]
    countries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in truth:
                countries[row["entity_id"]] = row["country"]
    out = {"candidate_file":PAIRS.name,"model_file":MODEL_PATH.name,"queries":len(ids),"candidate_pairs":len(df),"candidate_true_pairs":int(df.is_match.sum()),
           "threshold":THRESHOLD,"macro_f05":float(np.mean(scores)),
           "oracle_macro_f05":float(np.mean(oracle)),
           "complete_set_recall_nonempty":sum(truth[s].issubset(available[s]) for s in ids if truth[s])/max(1,sum(bool(truth[s]) for s in ids)),
           "tp":tp,"fp":fp,"fn":fn,"candidate_missed_true":sum(len(truth[s]-available[s]) for s in ids),
           "model_missed_available_true":sum(len(available[s]-predictions[s]) for s in ids),
           "pair_precision":tp/max(1,tp+fp),"pair_recall":tp/max(1,tp+fn),
           "singleton_count":len(singleton_ids),
           "singleton_accuracy":sum(not predictions[s] for s in singleton_ids)/max(1,len(singleton_ids)),
           "country_macro_f05":{country:float(np.mean([score for s,score in zip(ids,scores) if countries[s]==country])) for country in sorted(set(countries.values()))},
           "seconds":round(time.perf_counter()-tic,1)}
    METRICS.write_text(json.dumps(out,indent=2),encoding="utf-8")
    false_rows = df[(df.probability>=THRESHOLD)&(~df.is_match)].sort_values("probability",ascending=False).head(40)
    missed_rows = df[(df.probability<THRESHOLD)&(df.is_match)].sort_values("probability").head(40)
    ERRORS.write_text(json.dumps({"false_positive":false_rows.to_dict("records"),
        "missed_available_true":missed_rows.to_dict("records")},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2),flush=True)
    print("wrote",METRICS,"and local error sample",ERRORS,flush=True)


if __name__ == "__main__":
    main()
