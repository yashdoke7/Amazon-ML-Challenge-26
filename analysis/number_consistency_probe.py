"""Test precise address-number vetoes on separate development and validation samples."""

import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import FEATURE_NAMES, pair_features
from validation import f05_for_query
from calibrate_on_development import sampled_truth as development_truth
from evaluate_first_matcher import sampled_truth as validation_truth

NUMBER = re.compile(r"\d+")
MODEL = ROOT / "analysis" / "expanded_core_model.joblib"
THRESHOLD = 0.625


def first_number(value):
    match = NUMBER.search(value or "")
    return int(match.group()) if match else -1


def score(frame, selection, truth):
    predicted = {s: set() for s in truth}
    for s1, target in frame.loc[selection, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predicted[s1].add(target)
    return {"macro_f05": round(float(np.mean([f05_for_query(truth[s], predicted[s]) for s in truth])), 6),
            "tp": sum(len(predicted[s] & truth[s]) for s in truth),
            "fp": sum(len(predicted[s] - truth[s]) for s in truth),
            "fn": sum(len(truth[s] - predicted[s]) for s in truth)}


def run(label, filename, truth_fn, model):
    started = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("")
    truth = truth_fn()
    matrix = np.empty((len(frame), len(FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(frame.itertuples(index=False)):
        matrix[index] = pair_features(row.q_name, row.t_name, row.q_address,
                                      row.t_address, row.target_source)
    probability = model.predict_proba(matrix)[:, 1]
    selected = probability >= THRESHOLD
    qnum = np.array([first_number(value) for value in frame.q_address], dtype=np.int64)
    tnum = np.array([first_number(value) for value in frame.t_address], dtype=np.int64)
    different = (qnum >= 0) & (tnum >= 0) & (qnum != tnum)
    near = different & (np.abs(qnum - tnum) <= 10)
    exact_anchor = set(frame.loc[selected & (qnum == tnum) & (qnum >= 0), "s1_id"])
    anchor = frame.s1_id.isin(exact_anchor).to_numpy()
    name_similar = matrix[:, 0] >= 80
    address_similar = matrix[:, 8] >= 65
    options = {
        "baseline": np.zeros(len(frame), dtype=bool),
        "near_number_mismatch": near,
        "near_with_exact_anchor": near & anchor,
        "near_with_exact_anchor_us": near & anchor & (frame.country.to_numpy() == "US"),
        "near_with_exact_anchor_india": near & anchor & (frame.country.to_numpy() == "India"),
        "near_anchor_similar_name": near & anchor & name_similar,
        "near_anchor_similar_name_address": near & anchor & name_similar & address_similar,
        "any_mismatch_anchor_similar_name": different & anchor & name_similar,
    }
    results = {}
    for name, veto in options.items():
        removed = selected & veto
        results[name] = {"removed": int(removed.sum()),
                         "removed_tp": int((removed & frame.is_match.to_numpy()).sum()),
                         "removed_fp": int((removed & ~frame.is_match.to_numpy()).sum()),
                         **score(frame, selected & ~veto, truth)}
    print(label, json.dumps(results, indent=2), flush=True)
    return {"results": results, "seconds": round(time.perf_counter()-started, 1)}


def main():
    model = joblib.load(MODEL)
    assert list(model.feature_name_) == FEATURE_NAMES
    result = {"model": MODEL.name, "threshold": THRESHOLD,
              "development": run("development", "full_development_combined_pairs.parquet", development_truth, model),
              "validation": run("validation", "full_validation_combined_pairs.parquet", validation_truth, model)}
    (ROOT / "analysis" / "number_consistency_probe_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
