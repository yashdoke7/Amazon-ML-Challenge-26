"""Test multilingual name/address embeddings as extra matcher features on frozen data."""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from evaluate_first_matcher import sampled_truth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import FEATURE_NAMES, pair_features
from validation import f05_for_query

TRAIN = ROOT / "analysis" / "full_core_pairs.parquet"
VALIDATION = ROOT / "analysis" / "full_validation_combined_pairs.parquet"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
TEXT_COLUMNS = ["q_name", "t_name", "q_address", "t_address"]


def embedding_cosines(frame, encoder, left, right, label):
    combined = pd.concat([frame[left], frame[right]], ignore_index=True)
    codes, uniques = pd.factorize(combined, sort=False)
    count = len(frame)
    print(label, "unique texts", len(uniques), flush=True)
    vectors = np.empty((len(uniques), 384), dtype=np.float16)
    started = time.perf_counter()
    for first in range(0, len(uniques), 25_000):
        last = min(first + 25_000, len(uniques))
        vectors[first:last] = encoder.encode(uniques[first:last].tolist(), batch_size=128,
            device="cuda", show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
        if last % 100_000 < 25_000:
            print(label, "encoded", last, "seconds", round(time.perf_counter()-started, 1), flush=True)
    cosine = np.empty(count, dtype=np.float32)
    for first in range(0, count, 100_000):
        last = min(first + 100_000, count)
        a = vectors[codes[first:last]].astype(np.float32)
        b = vectors[codes[count+first:count+last]].astype(np.float32)
        cosine[first:last] = np.einsum("ij,ij->i", a, b)
    return cosine


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
    selected = np.flatnonzero(probabilities >= threshold)
    for s1, target in frame.iloc[selected][["s1_id", "target_id"]].itertuples(index=False, name=None):
        predictions[s1].add(target)
    ids = list(truth)
    return {"macro_f05": round(float(np.mean([f05_for_query(truth[s], predictions[s]) for s in ids])), 6),
            "tp": sum(len(predictions[s] & truth[s]) for s in ids),
            "fp": sum(len(predictions[s] - truth[s]) for s in ids),
            "fn": sum(len(truth[s] - predictions[s]) for s in ids),
            "singleton_accuracy": round(sum(not predictions[s] for s in ids if not truth[s]) /
                                        max(1, sum(not truth[s] for s in ids)), 5)}


def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    torch.set_num_threads(4)
    started = time.perf_counter()
    train = pd.read_parquet(TRAIN)
    validation = pd.read_parquet(VALIDATION)
    both = pd.concat([train, validation], ignore_index=True)
    both[TEXT_COLUMNS] = both[TEXT_COLUMNS].fillna("")
    print("pairs", len(train), len(validation), flush=True)
    encoder = SentenceTransformer(MODEL, device="cuda")
    cosine_name = embedding_cosines(both, encoder, "q_name", "t_name", "name")
    cosine_address = embedding_cosines(both, encoder, "q_address", "t_address", "address")
    del encoder
    torch.cuda.empty_cache()
    feature_names = FEATURE_NAMES + ["name_embedding_cosine", "address_embedding_cosine"]
    features = np.empty((len(both), len(feature_names)), dtype=np.float32)
    for index, row in enumerate(both.itertuples(index=False)):
        features[index, :len(FEATURE_NAMES)] = pair_features(row.q_name, row.t_name,
                                                             row.q_address, row.t_address, row.target_source)
        if index and index % 250_000 == 0:
            print("pair features", index, "seconds", round(time.perf_counter()-started, 1), flush=True)
    features[:, -2] = cosine_name
    features[:, -1] = cosine_address
    del cosine_name, cosine_address
    ntrain = len(train)
    labels = both.is_match.to_numpy(dtype=np.int8)
    train_ids = train.s1_id.to_numpy()
    internal_cal = np.array([int(s.split("-")[-1]) % 10 == 0 for s in train_ids])
    fit = (train.split.to_numpy() == "training") & ~internal_cal
    calibration = ((train.split.to_numpy() == "training") & internal_cal) | (train.split.to_numpy() == "development")
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=4, verbosity=-1, random_state=20260925)
    model.fit(features[:ntrain][fit], labels[:ntrain][fit], feature_name=feature_names,
              eval_set=[(features[:ntrain][calibration], labels[:ntrain][calibration])],
              eval_metric="binary_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
    cal_frame = train.loc[calibration].reset_index(drop=True)
    cal_probs = model.predict_proba(features[:ntrain][calibration])[:, 1]
    cal_truth = truth_for(cal_frame.s1_id.unique())
    thresholds = np.arange(0.35, 0.91, 0.025)
    best_t, best_cal = max(((float(t), score(cal_frame, cal_probs, t, cal_truth)) for t in thresholds),
                           key=lambda item: item[1]["macro_f05"])
    val_probs = model.predict_proba(features[ntrain:])[:, 1]
    val_truth = sampled_truth()
    assert set(validation.s1_id).issubset(val_truth)
    val_score = score(validation, val_probs, best_t, val_truth)
    out = {"model": MODEL, "embedding_license": "Apache-2.0", "train_pairs": len(train),
           "validation_pairs": len(validation), "best_iteration": model.best_iteration_,
           "calibration_threshold": round(best_t, 3), "calibration": best_cal,
           "frozen_validation": val_score,
           "feature_importance": dict(sorted(zip(feature_names, model.feature_importances_.tolist()),
                                             key=lambda item: item[1], reverse=True)),
           "seconds": round(time.perf_counter()-started, 1)}
    (ROOT / "analysis" / "embedding_feature_experiment_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model, ROOT / "analysis" / "embedding_feature_model.joblib")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
