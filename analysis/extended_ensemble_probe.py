"""Test whether character evidence complements the current 190k-query matcher.

Thresholds are chosen from development only; validation remains untouched.
The two models see identical broad candidate rows in each split.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from evaluate_generalized_model import score  # noqa: E402
from calibrate_on_development import sampled_truth as dev_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as val_truth  # noqa: E402
from extended_feature_probe import NAMES, features  # noqa: E402

GRID = (0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def prepare(filename, current, extended, started):
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    X = np.empty((len(frame), len(NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        X[i] = features(row)
        if i and i % 250_000 == 0:
            print(filename, "features", i, "seconds",
                  round(time.perf_counter()-started, 1), flush=True)
    return frame, current.predict_proba(X[:, :35])[:, 1], extended.predict_proba(X)[:, 1]


def thresholds(frame, truth, probabilities):
    sweeps = {t: score(frame, truth, probabilities, {"India": t, "other": t})
              for t in GRID}
    return {"India": max(GRID, key=lambda t: sweeps[t]["countries"]["India"]),
            "other": max(GRID, key=lambda t: sweeps[t]["countries"]["US"])}


def main():
    started = time.perf_counter()
    current = joblib.load(ROOT / "code" / "business_entity_resolution" / "generalized_model.joblib")
    extended = joblib.load(ROOT / "analysis" / "extended_feature_probe_model.joblib")
    assert list(extended.feature_name_) == NAMES
    dev = prepare("full_development_combined_pairs.parquet", current, extended, started)
    train_truth = dev_truth()
    settings = {}
    result = {}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        key = str(weight)
        p = (1-weight)*dev[1] + weight*dev[2]
        settings[key] = thresholds(dev[0], train_truth, p)
        result[key] = {"development": score(dev[0], train_truth, p, settings[key]),
                       "thresholds": settings[key]}
        print("development", key, result[key]["development"]["macro_f05"], flush=True)
    val = prepare("full_validation_combined_pairs.parquet", current, extended, started)
    frozen_truth = val_truth()
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        key = str(weight)
        p = (1-weight)*val[1] + weight*val[2]
        result[key]["validation"] = score(val[0], frozen_truth, p, settings[key])
        print("validation", key, result[key]["validation"]["macro_f05"], flush=True)
    result["seconds"] = round(time.perf_counter()-started, 1)
    (ROOT / "analysis" / "extended_ensemble_probe_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
