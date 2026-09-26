"""Test learned leading-address-number features without changing the frozen model."""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from number_features import NUMBER_FEATURE_NAMES, number_pair_features
from validation import f05_for_query
from evaluate_first_matcher import sampled_truth as validation_truth

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
TRAIN = ROOT / "analysis" / "full_core_pairs.parquet"
VALIDATION = ROOT / "analysis" / "full_validation_combined_pairs.parquet"


def truth_for(ids):
    wanted = set(ids)
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            s1 = row["source1_entity_id"]
            if s1 in wanted:
                truth[s1] = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    assert len(truth) == len(wanted)
    return truth


def score(frame, probabilities, threshold, truth):
    predictions = defaultdict(set)
    for s1, target in frame.loc[probabilities >= threshold, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predictions[s1].add(target)
    return {"macro_f05": round(float(np.mean([f05_for_query(truth[s], predictions[s]) for s in truth])), 6),
            "tp": sum(len(predictions[s] & truth[s]) for s in truth),
            "fp": sum(len(predictions[s] - truth[s]) for s in truth),
            "fn": sum(len(truth[s] - predictions[s]) for s in truth)}


def main():
    started = time.perf_counter()
    train = pd.read_parquet(TRAIN).fillna("")
    validation = pd.read_parquet(VALIDATION).fillna("")
    both = pd.concat([train, validation], ignore_index=True)
    names = NUMBER_FEATURE_NAMES
    matrix = np.empty((len(both), len(names)), dtype=np.float32)
    for i, row in enumerate(both.itertuples(index=False)):
        matrix[i] = number_pair_features(row.q_name, row.t_name,
                                         row.q_address, row.t_address, row.target_source)
        if i and i % 250_000 == 0:
            print("features", i, "seconds", round(time.perf_counter()-started, 1), flush=True)
    n = len(train)
    labels = train.is_match.to_numpy(dtype=np.int8)
    internal = np.array([int(s.split("-")[-1]) % 10 == 0 for s in train.s1_id])
    fit = (train.split.to_numpy() == "training") & ~internal
    calibration = ((train.split.to_numpy() == "training") & internal) | (train.split.to_numpy() == "development")
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=4, verbosity=-1, random_state=20260925)
    model.fit(matrix[:n][fit], labels[fit], feature_name=names,
              eval_set=[(matrix[:n][calibration], labels[calibration])],
              eval_metric="binary_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
    cal_frame = train.loc[calibration].reset_index(drop=True)
    cal_probs = model.predict_proba(matrix[:n][calibration])[:, 1]
    cal_truth = truth_for(cal_frame.s1_id.unique())
    thresholds = np.arange(0.35, 0.91, 0.025)
    threshold, calibration_score = max(((float(t), score(cal_frame, cal_probs, t, cal_truth))
                                        for t in thresholds), key=lambda value: value[1]["macro_f05"])
    val_probs = model.predict_proba(matrix[n:])[:, 1]
    val_score = score(validation, val_probs, threshold, validation_truth())
    out = {"train_pairs": n, "validation_pairs": len(validation),
           "best_iteration": model.best_iteration_, "threshold": round(threshold, 3),
           "calibration": calibration_score, "frozen_validation": val_score,
           "feature_importance": dict(sorted(zip(names, model.feature_importances_.tolist()),
                                             key=lambda pair: pair[1], reverse=True)),
           "seconds": round(time.perf_counter()-started, 1)}
    (ROOT / "analysis" / "first_number_feature_experiment_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model, ROOT / "analysis" / "first_number_feature_model.joblib")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
