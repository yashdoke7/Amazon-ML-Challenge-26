"""Calibrate 27-feature model on development, then score frozen validation."""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from number_features import NUMBER_FEATURE_NAMES, first_number, number_pair_features
from validation import f05_for_query
from calibrate_on_development import sampled_truth as development_truth
from evaluate_first_matcher import sampled_truth as validation_truth

MODEL = ROOT / "analysis" / "large_number_model.joblib"


def score(frame, selected, truth):
    predictions = defaultdict(set)
    for s, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predictions[s].add(target)
    return {"macro_f05": round(float(np.mean([f05_for_query(truth[s], predictions[s]) for s in truth])), 6),
            "tp": sum(len(predictions[s] & truth[s]) for s in truth),
            "fp": sum(len(predictions[s] - truth[s]) for s in truth),
            "fn": sum(len(truth[s] - predictions[s]) for s in truth),
            "singleton_accuracy": round(sum(not predictions[s] for s in truth if not truth[s]) /
                                        max(1, sum(not truth[s] for s in truth)), 5)}


def number_veto(frame, selected):
    qnum = np.array([first_number(value) for value in frame.q_address], dtype=np.int64)
    tnum = np.array([first_number(value) for value in frame.t_address], dtype=np.int64)
    anchors = set(frame.loc[selected & (qnum == tnum) & (qnum >= 0) &
                            (frame.country.to_numpy() == "US"), "s1_id"])
    remove = (selected & frame.s1_id.isin(anchors).to_numpy() & (tnum >= 0) &
              (np.abs(qnum - tnum) >= 1) & (np.abs(qnum - tnum) <= 10))
    return selected & ~remove


def evaluate(filename, truth_fn, model):
    started = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("")
    features = np.empty((len(frame), len(NUMBER_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        features[i] = number_pair_features(row.q_name, row.t_name,
                                           row.q_address, row.t_address, row.target_source)
    probabilities = model.predict_proba(features)[:, 1]
    truth = truth_fn()
    return frame, probabilities, truth, round(time.perf_counter()-started, 1)


def main():
    model = joblib.load(MODEL)
    assert list(model.feature_name_) == NUMBER_FEATURE_NAMES
    dev, dev_probs, dev_truth, dev_seconds = evaluate(
        "full_development_combined_pairs.parquet", development_truth, model)
    thresholds = np.arange(0.35, 0.91, 0.025)
    curve = [{"threshold": round(float(t), 3), **score(dev, dev_probs >= t, dev_truth)}
             for t in thresholds]
    best = max(curve, key=lambda item: item["macro_f05"])
    threshold = best["threshold"]
    val, val_probs, val_truth, val_seconds = evaluate(
        "full_validation_combined_pairs.parquet", validation_truth, model)
    dev_selected = dev_probs >= threshold
    val_selected = val_probs >= threshold
    out = {"model": MODEL.name, "best_iteration": model.best_iteration_,
           "development_best": best, "development_with_us_veto": score(dev, number_veto(dev, dev_selected), dev_truth),
           "frozen_validation": score(val, val_selected, val_truth),
           "frozen_validation_with_us_veto": score(val, number_veto(val, val_selected), val_truth),
           "development_seconds": dev_seconds, "validation_seconds": val_seconds}
    (ROOT / "analysis" / "large_number_model_evaluation_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
