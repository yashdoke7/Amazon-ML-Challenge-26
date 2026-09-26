"""Test a locally trained specialist for candidates lacking a target address."""

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
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from validation import f05_for_query  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

BASE = ROOT / "code" / "business_entity_resolution" / "generalized_model.joblib"
MODEL = ROOT / "analysis" / "blank_address_specialist_model.joblib"
GRID = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25,
        0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def matrix(frame):
    features = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        features[i] = generalized_pair_features(
            row.q_name, row.t_name, row.q_address, row.t_address, row.target_source)
    return features


def prepare(filename, truth_fn, base, specialist):
    started = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    features = matrix(frame)
    blank = frame.t_address.to_numpy() == ""
    base_prob = base.predict_proba(features)[:, 1]
    specialist_prob = specialist.predict_proba(features[blank])[:, 1]
    return frame, truth_fn(), blank, base_prob, specialist_prob, round(time.perf_counter()-started, 1)


def score(data, specialist_threshold=None):
    frame, truth, blank, base_prob, specialist_prob, _ = data
    base_threshold = np.where(frame.country.to_numpy() == "India", 0.65, 0.75)
    selected = base_prob >= base_threshold
    if specialist_threshold is not None:
        selected[blank] = specialist_prob >= specialist_threshold
    predictions = defaultdict(set)
    for q, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predictions[q].add(target)
    labels = frame.is_match.to_numpy(dtype=bool)
    values = {q: f05_for_query(ids, predictions[q]) for q, ids in truth.items()}
    countries = frame.groupby("country").s1_id.unique().to_dict()
    return {"macro_f05": float(np.mean(list(values.values()))),
            "tp": int((selected & labels).sum()), "fp": int((selected & ~labels).sum()),
            "blank_tp": int((selected & blank & labels).sum()),
            "blank_fp": int((selected & blank & ~labels).sum()),
            "blank_reachable_positives": int((blank & labels).sum()),
            "country_macro_f05": {c: float(np.mean([values[q] for q in ids]))
                                  for c, ids in countries.items()}}


def main():
    started = time.perf_counter()
    train = pd.read_parquet(ROOT / "analysis" / "full_combined_pairs.parquet").fillna("")
    train = train.loc[train.t_address == ""].reset_index(drop=True)
    features = matrix(train)
    labels = train.is_match.to_numpy(dtype=np.int8)
    owned = train.split.to_numpy() == "training"
    internal = np.fromiter((int(q.split("-")[-1]) % 10 == 0 for q in train.s1_id),
                           dtype=bool, count=len(train))
    fit = owned & ~internal
    cal = owned & internal
    print("blank train rows", len(train), "positives", int(labels.sum()),
          "fit", int(fit.sum()), "cal", int(cal.sum()),
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    specialist = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=4, verbosity=-1, random_state=20260926)
    specialist.fit(features[fit], labels[fit], feature_name=GENERALIZED_FEATURE_NAMES,
                   eval_set=[(features[cal], labels[cal])], eval_metric="binary_logloss",
                   callbacks=[lgb.early_stopping(30, verbose=False)])
    joblib.dump(specialist, MODEL)
    print("best iteration", specialist.best_iteration_,
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    del train, features, labels
    base = joblib.load(BASE)
    dev = prepare("full_development_combined_pairs.parquet", development_truth, base, specialist)
    sweep = {str(t): score(dev, t) for t in GRID}
    chosen = max(GRID, key=lambda t: sweep[str(t)]["macro_f05"])
    val = prepare("full_validation_combined_pairs.parquet", validation_truth, base, specialist)
    result = {"train_blank_pairs": int(owned.sum()), "best_iteration": int(specialist.best_iteration_),
              "chosen_on_development": chosen,
              "development_baseline": score(dev), "development_sweep": sweep,
              "validation_baseline": score(val), "validation_chosen": score(val, chosen),
              "seconds": {"total": round(time.perf_counter()-started, 1),
                          "development": dev[5], "validation": val[5]}}
    destination = ROOT / "analysis" / "blank_address_specialist_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
