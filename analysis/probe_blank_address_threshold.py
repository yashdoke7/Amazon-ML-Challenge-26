"""Evaluate a conditional threshold for targets with no supplied address.

Choose on the fixed development sample, then report the unchanged validation
sample once. No test labels or target text outside supplied TSVs are used.
"""

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
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from validation import f05_for_query  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

MODEL = ROOT / "code" / "business_entity_resolution" / "generalized_model.joblib"
GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]


def prepare(filename, truth_fn, model):
    started = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    features = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        features[i] = generalized_pair_features(
            row.q_name, row.t_name, row.q_address, row.t_address, row.target_source)
    return frame, truth_fn(), model.predict_proba(features)[:, 1], round(time.perf_counter()-started, 1)


def score(frame, truth, probabilities, blank_threshold):
    blank = frame.t_address.to_numpy() == ""
    base = np.where(frame.country.to_numpy() == "India", 0.65, 0.75)
    thresholds = base if blank_threshold is None else np.where(blank, blank_threshold, base)
    selected = probabilities >= thresholds
    predicted = defaultdict(set)
    for query, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predicted[query].add(target)
    labels = frame.is_match.to_numpy(dtype=bool)
    value = {q: f05_for_query(ids, predicted[q]) for q, ids in truth.items()}
    countries = frame.groupby("country").s1_id.unique().to_dict()
    return {"macro_f05": float(np.mean(list(value.values()))),
            "tp": int((selected & labels).sum()),
            "fp": int((selected & ~labels).sum()),
            "blank_selected_tp": int((selected & blank & labels).sum()),
            "blank_selected_fp": int((selected & blank & ~labels).sum()),
            "blank_reachable_positives": int((blank & labels).sum()),
            "country_macro_f05": {country: float(np.mean([value[q] for q in queries]))
                                  for country, queries in countries.items()},
            "singleton_accuracy": float(sum(not predicted[q] for q, ids in truth.items() if not ids) /
                                        max(1, sum(not ids for ids in truth.values())))}


def main():
    model = joblib.load(MODEL)
    dev = prepare("full_development_combined_pairs.parquet", development_truth, model)
    baseline = score(dev[0], dev[1], dev[2], None)
    sweep = {str(t): score(dev[0], dev[1], dev[2], t) for t in GRID}
    chosen = max(GRID, key=lambda t: sweep[str(t)]["macro_f05"])
    val = prepare("full_validation_combined_pairs.parquet", validation_truth, model)
    result = {"chosen_on_development": chosen, "development_baseline": baseline,
              "development_sweep": sweep, "validation_baseline": score(val[0], val[1], val[2], None),
              "validation_chosen": score(val[0], val[1], val[2], chosen),
              "seconds": {"development": dev[3], "validation": val[3]}}
    destination = ROOT / "analysis" / "blank_address_threshold_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
