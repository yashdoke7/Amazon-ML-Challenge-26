"""Evaluate a hard-negative model against the frozen 35-feature reference.

Choose thresholds using the development sample only. The separate validation
sample is read once with those thresholds and never used to tune them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from evaluate_generalized_model import score  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

BASE = ROOT / "code" / "business_entity_resolution" / "generalized_model.joblib"
NEW = ROOT / "analysis" / "hard_negative_generalized_model.joblib"
GRID = [round(x, 2) for x in np.arange(0.35, 0.951, 0.05)]


def prepare(filename, truth_fn, models):
    start = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    features = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        features[i] = generalized_pair_features(
            row.q_name, row.t_name, row.q_address, row.t_address, row.target_source)
    scores = {name: model.predict_proba(features)[:, 1] for name, model in models.items()}
    return frame, truth_fn(), scores, round(time.perf_counter() - start, 1)


def choose(frame, truth, probabilities):
    candidates = []
    for threshold in GRID:
        settings = {"India": threshold, "other": threshold}
        result = score(frame, truth, probabilities, settings)
        candidates.append((threshold, result))
    india = max(candidates, key=lambda item: item[1]["countries"]["India"])[0]
    us = max(candidates, key=lambda item: item[1]["countries"]["US"])[0]
    return {"India": india, "other": us}, {
        str(t): {"macro_f05": v["macro_f05"], "countries": v["countries"]}
        for t, v in candidates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=NEW)
    parser.add_argument("--training-summary", type=Path,
                        default=ROOT / "analysis" / "hard_negative_generalized_training_results.json")
    parser.add_argument("--result", type=Path,
                        default=ROOT / "analysis" / "hard_negative_generalized_evaluation_results.json")
    args = parser.parse_args()
    models = {"reference": joblib.load(BASE), "hard_negative": joblib.load(args.model)}
    for model in models.values():
        assert list(model.feature_name_) == GENERALIZED_FEATURE_NAMES
    dev = prepare("full_development_combined_pairs.parquet", development_truth, models)
    chosen, sweep = choose(dev[0], dev[1], dev[2]["hard_negative"])
    val = prepare("full_validation_combined_pairs.parquet", validation_truth, models)
    baseline_threshold = {"India": 0.65, "other": 0.75}
    result = {
        "training_summary": json.loads(args.training_summary.read_text()),
        "thresholds_selected_on_development_only": chosen,
        "development_sweep": sweep,
        "development_reference": score(dev[0], dev[1], dev[2]["reference"], baseline_threshold),
        "development_new": score(dev[0], dev[1], dev[2]["hard_negative"], chosen),
        "validation_reference": score(val[0], val[1], val[2]["reference"], baseline_threshold),
        "validation_new": score(val[0], val[1], val[2]["hard_negative"], chosen),
        "seconds": {"development": dev[3], "validation": val[3]},
    }
    destination = args.result
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
