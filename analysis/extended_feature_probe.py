"""Compare cheap character/partial-match evidence with a same-data 35-feature model.

This is an isolated feasibility probe. It uses training-owned rows to fit both
models, development to choose thresholds, and frozen validation only to report.
It does not change the packaged submission or use external business data.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features, ascii_normalize  # noqa: E402
from evaluate_generalized_model import score  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

EXTRA = [
    "ascii_compact_name_ratio", "ascii_name_partial", "ascii_address_partial",
    "name_char3_jaccard", "name_char3_containment",
    "address_char3_jaccard", "address_char3_containment",
]
NAMES = GENERALIZED_FEATURE_NAMES + EXTRA


def grams(text):
    text = text.replace(" ", "")
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text)-2)}


def overlaps(a, b):
    if not a or not b:
        return 0.0, 0.0
    n = len(a & b)
    return n / len(a | b), n / min(len(a), len(b))


def features(row):
    base = generalized_pair_features(row.q_name, row.t_name, row.q_address,
                                     row.t_address, row.target_source)
    qn, tn = ascii_normalize(row.q_name), ascii_normalize(row.t_name)
    qa, ta = ascii_normalize(row.q_address), ascii_normalize(row.t_address)
    n_j, n_c = overlaps(grams(qn), grams(tn))
    a_j, a_c = overlaps(grams(qa), grams(ta))
    return (*base, fuzz.ratio(qn.replace(" ", ""), tn.replace(" ", "")),
            fuzz.partial_ratio(qn, tn), fuzz.partial_ratio(qa, ta),
            n_j, n_c, a_j, a_c)


def matrix(frame, started):
    out = np.empty((len(frame), len(NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        out[i] = features(row)
        if i and i % 250_000 == 0:
            print("features", i, "of", len(frame),
                  "seconds", round(time.perf_counter()-started, 1), flush=True)
    return out


def choose(frame, truth, probabilities):
    trials = []
    for india in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
        for other in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
            setting = {"India": india, "other": other}
            trials.append((score(frame, truth, probabilities, setting)["macro_f05"], setting))
    return max(trials, key=lambda x: x[0])[1]


def fit(X, y, mask, cal, names):
    model = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.04, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=4, verbosity=-1, random_state=20260927)
    model.fit(X[mask], y[mask], feature_name=names,
              eval_set=[(X[cal], y[cal])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(40, verbose=False)])
    return model


def main():
    started = time.perf_counter()
    train = pd.read_parquet(ROOT / "analysis" / "full_combined_pairs.parquet").fillna("")
    dev = pd.read_parquet(ROOT / "analysis" / "full_development_combined_pairs.parquet").fillna("")
    val = pd.read_parquet(ROOT / "analysis" / "full_validation_combined_pairs.parquet").fillna("")
    train = train.loc[train.split == "training"].reset_index(drop=True)
    X = matrix(train, started)
    y = train.is_match.to_numpy(dtype=np.int8)
    cal = np.fromiter((int(s.split("-")[-1]) % 10 == 0 for s in train.s1_id),
                      dtype=bool, count=len(train))
    print("fit baseline", len(train), "pairs", int(y.sum()), "positives", flush=True)
    baseline = fit(X[:, :35], y, ~cal, cal, GENERALIZED_FEATURE_NAMES)
    print("fit extended", round(time.perf_counter()-started, 1), flush=True)
    extended = fit(X, y, ~cal, cal, NAMES)
    del X, train
    results = {"training_pairs": len(y), "training_positives": int(y.sum()),
               "baseline_best_iteration": int(baseline.best_iteration_),
               "extended_best_iteration": int(extended.best_iteration_)}
    for label, frame, truth in (("development", dev, development_truth()),
                                ("validation", val, validation_truth())):
        X = matrix(frame, started)
        for name, model, mat in (("baseline", baseline, X[:, :35]),
                                 ("extended", extended, X)):
            p = model.predict_proba(mat)[:, 1]
            if label == "development":
                results[name + "_thresholds"] = choose(frame, truth, p)
            settings = results[name + "_thresholds"]
            results[label + "_" + name] = score(frame, truth, p, settings)
        del X
        print(label, {name: results[label + "_" + name]["macro_f05"]
                      for name in ("baseline", "extended")}, flush=True)
    results["extended_importance"] = dict(sorted(
        zip(NAMES, extended.feature_importances_.tolist()),
        key=lambda item: -item[1]))
    results["seconds"] = round(time.perf_counter()-started, 1)
    (ROOT / "analysis" / "extended_feature_probe_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    joblib.dump(extended, ROOT / "analysis" / "extended_feature_probe_model.joblib")
    print(json.dumps({"scores": {k: v["macro_f05"] for k, v in results.items()
                                if k.startswith(("development_", "validation_"))
                                and isinstance(v, dict) and "macro_f05" in v},
                      "seconds": results["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
