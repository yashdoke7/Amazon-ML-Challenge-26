"""Compare the Unicode-aware matcher with the frozen 27-feature model.

Choose any new thresholds on the development sample, then report the unchanged
validation sample once. These are raw pair decisions; full postprocessing and
French transfer remain separate gates.
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


def score(frame, truth, scores, thresholds):
    selected = scores >= np.where(frame.country.to_numpy() == "India",
                                  thresholds["India"], thresholds["other"])
    predicted = defaultdict(set)
    for query, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predicted[query].add(target)
    countries = frame.groupby("country").s1_id.unique().to_dict()
    out = {"macro_f05": float(np.mean([f05_for_query(ids, predicted[q])
                                         for q, ids in truth.items()])),
           "selected": int(selected.sum()),
           "tp": int((selected & frame.is_match.to_numpy(dtype=bool)).sum()),
           "fp": int((selected & ~frame.is_match.to_numpy(dtype=bool)).sum()),
           "countries": {country: float(np.mean([
               f05_for_query(truth[q], predicted[q]) for q in queries
           ])) for country, queries in countries.items()}}
    out["singleton_accuracy"] = sum(not predicted[q] for q, ids in truth.items() if not ids) / max(
        1, sum(not ids for ids in truth.values()))
    cross = np.fromiter((any(ord(c) > 127 for c in a) != any(ord(c) > 127 for c in b)
                         for a, b in zip(frame.q_name, frame.t_name)),
                        dtype=bool, count=len(frame))
    positives = frame.is_match.to_numpy(dtype=bool)
    out["cross_script_positive_recall"] = float((selected & cross & positives).sum() /
                                                 max(1, (cross & positives).sum()))
    return out


def prepare(filename, truth_fn, new_model, old_model):
    started = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    features = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        features[i] = generalized_pair_features(
            row.q_name, row.t_name, row.q_address, row.t_address, row.target_source)
    return (frame, truth_fn(), new_model.predict_proba(features)[:, 1],
            old_model.predict_proba(features[:, :27])[:, 1],
            round(time.perf_counter() - started, 1))


def main():
    new_model = joblib.load(ROOT / "analysis" / "generalized_model.joblib")
    old_model = joblib.load(ROOT / "code" / "business_entity_resolution" / "number_model.joblib")
    assert list(new_model.feature_name_) == GENERALIZED_FEATURE_NAMES
    dev = prepare("full_development_combined_pairs.parquet", development_truth,
                  new_model, old_model)
    baseline = {"India": 0.65, "other": 0.75}
    trials = []
    for india in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        for other in (0.65, 0.7, 0.75, 0.8, 0.85):
            setting = {"India": india, "other": other}
            trials.append((score(dev[0], dev[1], dev[2], setting)["macro_f05"], setting))
    _, chosen = max(trials, key=lambda item: item[0])
    val = prepare("full_validation_combined_pairs.parquet", validation_truth,
                  new_model, old_model)
    result = {
        "new_model": "generalized_model.joblib",
        "new_best_iteration": int(new_model.best_iteration_),
        "development_selected_thresholds": chosen,
        "development_old_frozen": score(dev[0], dev[1], dev[3], baseline),
        "development_new_at_old_thresholds": score(dev[0], dev[1], dev[2], baseline),
        "development_new_chosen": score(dev[0], dev[1], dev[2], chosen),
        "validation_old_frozen": score(val[0], val[1], val[3], baseline),
        "validation_new_at_old_thresholds": score(val[0], val[1], val[2], baseline),
        "validation_new_chosen": score(val[0], val[1], val[2], chosen),
        "seconds": {"development": dev[4], "validation": val[4]},
    }
    (ROOT / "analysis" / "generalized_model_evaluation_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
