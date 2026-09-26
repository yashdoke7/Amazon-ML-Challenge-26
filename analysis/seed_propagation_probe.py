"""Test whether high-confidence target links can rescue rejected candidates."""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import FEATURE_NAMES, pair_features
from validation import f05_for_query
from calibrate_on_development import sampled_truth as development_truth
from evaluate_first_matcher import sampled_truth as validation_truth

MODEL = ROOT / "analysis" / "expanded_core_model.joblib"
THRESHOLD = 0.625


def score(frame, selected, truth):
    groups = {s: set() for s in truth}
    for s, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        groups[s].add(target)
    return {"macro_f05": round(float(np.mean([f05_for_query(truth[s], groups[s]) for s in truth])), 6),
            "tp": sum(len(groups[s] & truth[s]) for s in truth),
            "fp": sum(len(groups[s] - truth[s]) for s in truth),
            "fn": sum(len(truth[s] - groups[s]) for s in truth)}


def run(label, filename, truth_fn, model):
    started = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("")
    truth = truth_fn()
    features = np.empty((len(frame), len(FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(frame.itertuples(index=False)):
        features[index] = pair_features(row.q_name, row.t_name, row.q_address,
                                        row.t_address, row.target_source)
    probability = model.predict_proba(features)[:, 1]
    base = probability >= THRESHOLD
    seeds = defaultdict(list)
    for s, name, address in frame.loc[base, ["s1_id", "t_name", "t_address"]].itertuples(index=False, name=None):
        seeds[s].append((name, address))
    viable = np.flatnonzero((~base) & (probability >= 0.05) & frame.s1_id.isin(seeds).to_numpy())
    name_sim = np.zeros(len(frame), dtype=np.float32)
    name_set_sim = np.zeros(len(frame), dtype=np.float32)
    address_sim = np.zeros(len(frame), dtype=np.float32)
    for i in viable:
        row = frame.iloc[i]
        comparisons = seeds[row.s1_id]
        name_sim[i] = max(fuzz.ratio(row.t_name, name) for name, _ in comparisons)
        name_set_sim[i] = max(fuzz.token_set_ratio(row.t_name, name) for name, _ in comparisons)
        address_sim[i] = max(fuzz.ratio(row.t_address, address) for _, address in comparisons)
    options = {
        "baseline": np.zeros(len(frame), dtype=bool),
        "strict_name_address": (probability >= 0.1) & (name_sim >= 90) & (address_sim >= 90),
        "name_focused": (probability >= 0.05) & (name_sim >= 95) & (address_sim >= 80),
        "address_focused": (probability >= 0.05) & (name_sim >= 80) & (address_sim >= 95),
        "higher_model_floor": (probability >= 0.2) & (name_sim >= 80) & (address_sim >= 90),
        "token_name_address": (probability >= 0.1) & (name_set_sim >= 90) & (address_sim >= 90),
    }
    out = {}
    for name, added in options.items():
        added &= ~base
        out[name] = {"added": int(added.sum()),
                     "added_tp": int((added & frame.is_match.to_numpy()).sum()),
                     "added_fp": int((added & ~frame.is_match.to_numpy()).sum()),
                     **score(frame, base | added, truth)}
    result = {"candidate_pairs": len(frame), "base_selected": int(base.sum()),
              "viable_unselected": len(viable), "results": out,
              "seconds": round(time.perf_counter()-started, 1)}
    print(label, json.dumps(result, indent=2), flush=True)
    return result


def main():
    model = joblib.load(MODEL)
    assert list(model.feature_name_) == FEATURE_NAMES
    out = {"model": MODEL.name, "threshold": THRESHOLD,
           "development": run("development", "full_development_combined_pairs.parquet", development_truth, model),
           "validation": run("validation", "full_validation_combined_pairs.parquet", validation_truth, model)}
    (ROOT / "analysis" / "seed_propagation_probe_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
