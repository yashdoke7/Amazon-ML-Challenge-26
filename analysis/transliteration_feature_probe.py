"""Measure a local ASCII-transliteration rescue on existing candidate pairs.

Install the analysis-only dependency with `pip install anyascii==0.3.3`.
This does not test newly retrieved candidates or change the submission pipeline.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from validation import f05_for_query  # noqa: E402
from evaluate_large_number_model import evaluate  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402


def has_indic(text):
    return any(0x0900 <= ord(c) <= 0x0D7F for c in text)


def prepare(filename, truth_fn, model):
    frame, probs, truth, seconds = evaluate(filename, truth_fn, model)
    frame = frame.reset_index(drop=True)
    base = probs >= np.where(frame.country.to_numpy() == "India", 0.65, 0.75)
    cross = np.fromiter((has_indic(x) != has_indic(y)
                         for x, y in zip(frame.q_name, frame.t_name)),
                        dtype=bool, count=len(frame))
    cross &= frame.country.to_numpy() == "India"
    ratios = np.zeros(len(frame), dtype=np.float32)
    addresses = np.zeros(len(frame), dtype=np.float32)
    for i in np.flatnonzero(cross & ~base):
        row = frame.iloc[i]
        ratios[i] = fuzz.ratio(anyascii(row.q_name).casefold(),
                               anyascii(row.t_name).casefold())
        addresses[i] = fuzz.ratio(row.q_address.casefold(),
                                  row.t_address.casefold())
    return frame, probs, truth, base, cross, ratios, addresses, seconds


def score(frame, mask, truth):
    groups = defaultdict(set)
    for query, target in frame.loc[mask, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        groups[query].add(target)
    tp = sum(len(groups[q] & true_ids) for q, true_ids in truth.items())
    fp = sum(len(groups[q] - true_ids) for q, true_ids in truth.items())
    return {"macro_f05": round(float(np.mean([
        f05_for_query(true_ids, groups[q]) for q, true_ids in truth.items()])), 6),
        "tp": tp, "fp": fp, "fn": sum(map(len, truth.values())) - tp}


def grid(data):
    frame, probs, truth, base, cross, ratios, addresses, _ = data
    base_score = score(frame, base, truth)
    trials = []
    for name_min in (65, 70, 75, 80, 85):
        for address_min in (0, 50, 65, 80):
            for probability_min in (0.01, 0.05, 0.10, 0.20):
                added = cross & ~base & (ratios >= name_min) & \
                    (addresses >= address_min) & (probs >= probability_min)
                result = score(frame, base | added, truth)
                result.update({"name_min": name_min, "address_min": address_min,
                               "probability_min": probability_min,
                               "added": int(added.sum()),
                               "added_tp": result["tp"] - base_score["tp"],
                               "added_fp": result["fp"] - base_score["fp"]})
                trials.append(result)
    return base_score, sorted(trials, key=lambda row: row["macro_f05"], reverse=True)


def apply(data, setting):
    frame, probs, truth, base, cross, ratios, addresses, _ = data
    added = cross & ~base & (ratios >= setting["name_min"]) & \
        (addresses >= setting["address_min"]) & \
        (probs >= setting["probability_min"])
    result = score(frame, base | added, truth)
    result["added"] = int(added.sum())
    return result


def main():
    model = joblib.load(ROOT / "code" / "business_entity_resolution" / "number_model.joblib")
    dev = prepare("full_development_combined_pairs.parquet", development_truth, model)
    val = prepare("full_validation_combined_pairs.parquet", validation_truth, model)
    dev_base, trials = grid(dev)
    val_base = score(val[0], val[3], val[2])
    best = trials[0]
    result = {"dependency": "anyascii==0.3.3 (ISC, analysis only)",
              "development_base": dev_base, "development_best": best,
              "development_top_10": trials[:10],
              "validation_base": val_base,
              "validation_chosen": apply(val, best),
              "cross_candidates": {"development": int(dev[4].sum()),
                                   "validation": int(val[4].sum())}}
    destination = ROOT / "analysis" / "transliteration_feature_probe_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
