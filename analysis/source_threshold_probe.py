"""Check source-specific thresholds for the frozen 27-feature matcher.

Select on the development sample and report unchanged validation performance.
This is an analysis probe; it does not alter the running full-test rescore.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from validation import f05_for_query  # noqa: E402
from evaluate_large_number_model import evaluate  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

MODEL = ROOT / "code" / "business_entity_resolution" / "number_model.joblib"
GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
BASE = {"US:S2": 0.75, "US:S3": 0.75, "India:S2": 0.65, "India:S3": 0.65}


def prepare(frame, probs, truth):
    frame = frame.reset_index(drop=True)
    keys = (frame.country + ":" + frame.target_source).to_numpy()
    s1 = frame.s1_id.to_numpy()
    target = frame.target_id.to_numpy()
    assert set(keys) == set(BASE)
    return (keys, s1, target, probs, truth)


def score(data, thresholds):
    keys, s1, target, probs, truth = data
    cutoff = np.array([thresholds[k] for k in keys], dtype=np.float32)
    selected = np.flatnonzero(probs >= cutoff)
    groups = defaultdict(set)
    for i in selected:
        groups[s1[i]].add(target[i])
    values = [f05_for_query(true_ids, groups[q]) for q, true_ids in truth.items()]
    tp = sum(len(true_ids & groups[q]) for q, true_ids in truth.items())
    fp = sum(len(groups[q] - true_ids) for q, true_ids in truth.items())
    return {"macro_f05": round(float(np.mean(values)), 6), "tp": tp,
            "fp": fp, "fn": sum(map(len, truth.values())) - tp,
            "selected": len(selected)}


def main():
    model = joblib.load(MODEL)
    dev = prepare(*evaluate("full_development_combined_pairs.parquet", development_truth, model)[:3])
    val = prepare(*evaluate("full_validation_combined_pairs.parquet", validation_truth, model)[:3])
    baseline = {"development": score(dev, BASE), "validation": score(val, BASE)}
    single = {}
    for key in BASE:
        candidates = []
        for threshold in GRID:
            trial = dict(BASE, **{key: threshold})
            candidates.append({"threshold": threshold, **score(dev, trial)})
        best = max(candidates, key=lambda x: x["macro_f05"])
        trial = dict(BASE, **{key: best["threshold"]})
        single[key] = {"chosen": best, "validation": score(val, trial),
                       "curve": candidates}
        print(key, best, "validation", single[key]["validation"], flush=True)
    combined = dict(BASE)
    for key, result in single.items():
        combined[key] = result["chosen"]["threshold"]
    result = {"baseline": baseline, "source_single_changes": single,
              "combined_thresholds": combined,
              "combined_development": score(dev, combined),
              "combined_validation": score(val, combined)}
    destination = ROOT / "analysis" / "source_threshold_probe_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RESULT", destination, result["combined_development"],
          result["combined_validation"], flush=True)


if __name__ == "__main__":
    main()
