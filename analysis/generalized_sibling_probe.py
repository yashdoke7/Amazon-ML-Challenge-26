"""Test whether selected sibling records rescue weak candidate links.

Choose the rule on the 2k development sample and evaluate once on the frozen
2k validation sample. This is a bounded local probe, not a final rule.
"""

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz

from evaluate_generalized_model import prepare, score
from calibrate_on_development import sampled_truth as development_truth
from evaluate_first_matcher import sampled_truth as validation_truth

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {"India": 0.8, "other": 0.8}


def attach_context(frame, probs):
    base = probs >= 0.8
    anchors = defaultdict(list)
    for q, target, name, address in frame.loc[
            base, ["s1_id", "target_id", "t_name", "t_address"]].itertuples(index=False, name=None):
        anchors[q].append((target, name, address))
    viable = np.flatnonzero(~base & (probs >= 0.02) & frame.s1_id.isin(anchors).to_numpy())
    name = np.zeros(len(frame), dtype=np.float32)
    ascii_name = np.zeros(len(frame), dtype=np.float32)
    address = np.zeros(len(frame), dtype=np.float32)
    address_set = np.zeros(len(frame), dtype=np.float32)
    for i in viable:
        row = frame.iloc[i]
        sibling = [(n, a) for t, n, a in anchors[row.s1_id] if t != row.target_id]
        if not sibling:
            continue
        name[i] = max(fuzz.ratio(row.t_name, n) for n, _ in sibling)
        ascii_name[i] = max(fuzz.ratio(anyascii(row.t_name), anyascii(n))
                            for n, _ in sibling)
        address[i] = max(fuzz.ratio(row.t_address, a) for _, a in sibling)
        address_set[i] = max(fuzz.token_set_ratio(row.t_address, a) for _, a in sibling)
    return base, viable, name, ascii_name, address, address_set


def options(data):
    frame, truth, probs, _, _ = data
    base, viable, name, ascii_name, address, address_set = attach_context(frame, probs)
    masks = {
        "strict_name_address": (probs >= 0.05) & (name >= 90) & (address >= 90),
        "strong_name_address": (probs >= 0.10) & (name >= 95) & (address >= 80),
        "strong_address_name": (probs >= 0.10) & (address >= 95) & (name >= 75),
        "translit_name_address": (probs >= 0.10) & (ascii_name >= 90) & (address >= 80),
        "address_set_name": (probs >= 0.20) & (address_set >= 95) & (name >= 80),
    }
    baseline = score(frame, truth, probs, THRESHOLDS)
    results = {"baseline": baseline, "viable": len(viable)}
    for key, add in masks.items():
        chosen = base | (add & ~base)
        revised = np.where(chosen, 1.0, 0.0)
        metric = score(frame, truth, revised, {"India": 0.5, "other": 0.5})
        metric["added"] = int((add & ~base).sum())
        metric["added_tp"] = metric["tp"] - baseline["tp"]
        metric["added_fp"] = metric["fp"] - baseline["fp"]
        results[key] = metric
    return results


def main():
    model = joblib.load(ROOT / "analysis" / "generalized_model.joblib")
    old = joblib.load(ROOT / "code" / "business_entity_resolution" / "number_model.joblib")
    dev = options(prepare("full_development_combined_pairs.parquet", development_truth, model, old))
    val = options(prepare("full_validation_combined_pairs.parquet", validation_truth, model, old))
    chosen = max((key for key in dev if key not in ("baseline", "viable")),
                 key=lambda key: dev[key]["macro_f05"])
    result = {"chosen_on_development": chosen, "development": dev,
              "validation_chosen": val[chosen], "validation_baseline": val["baseline"],
              "validation_all_diagnostic": val}
    (ROOT / "analysis" / "generalized_sibling_probe_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
