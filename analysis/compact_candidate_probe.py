"""Measure honest pre-model top-k candidate budgets on frozen labeled pools.

Ranks each existing candidate by raw name and address Jaro-Winkler similarity,
then keeps independent quotas. This is a feasibility check before changing the
full retrieval pipeline. It uses labels only for evaluation, never selection.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from calibrate_on_development import sampled_truth as development_truth
from evaluate_first_matcher import sampled_truth as validation_truth
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from validation import f05_for_query  # noqa: E402

OPTIONS = [(15, 15), (20, 20), (25, 15), (15, 25),
           (25, 20), (20, 25), (30, 20), (20, 30)]


def evaluate(filename, truth_fn):
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("")
    truth = truth_fn()
    names = np.fromiter((JaroWinkler.normalized_similarity(x.casefold(), y.casefold())
                         for x, y in zip(frame.q_name, frame.t_name)),
                        dtype=np.float32, count=len(frame))
    addresses = np.fromiter((JaroWinkler.normalized_similarity(x.casefold(), y.casefold())
                             for x, y in zip(frame.q_address, frame.t_address)),
                            dtype=np.float32, count=len(frame))
    frame["name_sim"] = names
    frame["address_sim"] = addresses
    name_order = frame.sort_values(["s1_id", "name_sim", "target_id"],
                                   ascending=[True, False, True], kind="stable").index
    address_order = frame.sort_values(["s1_id", "address_sim", "target_id"],
                                      ascending=[True, False, True], kind="stable").index
    frame.loc[name_order, "name_rank"] = frame.loc[name_order].groupby("s1_id", sort=False).cumcount().to_numpy() + 1
    frame.loc[address_order, "address_rank"] = frame.loc[address_order].groupby("s1_id", sort=False).cumcount().to_numpy() + 1
    result = {"queries": len(truth), "input_pairs": len(frame), "options": {}}
    for name_k, address_k in OPTIONS:
        selected = (frame.name_rank.to_numpy() <= name_k) | (frame.address_rank.to_numpy() <= address_k)
        kept = frame.loc[selected, ["s1_id", "target_id", "is_match"]]
        true_kept = int(kept.is_match.sum())
        groups = defaultdict(set)
        for s1, target in kept.loc[kept.is_match, ["s1_id", "target_id"]].itertuples(index=False, name=None):
            groups[s1].add(target)
        result["options"][f"name{name_k}_address{address_k}"] = {
            "candidate_pairs": int(selected.sum()),
            "pairs_per_query": round(float(selected.sum() / len(truth)), 2),
            "positive_links": true_kept,
            "edge_recall": round(true_kept / sum(map(len, truth.values())), 6),
            "oracle_macro_f05": round(float(np.mean([
                f05_for_query(true_ids, groups[s1]) for s1, true_ids in truth.items()])), 6),
            "estimated_test_zip_mb_at_5p7_bytes_per_pair": round(
                (selected.sum() / len(truth)) * 1_732_544 * 5.7 / 1e6, 1),
        }
    return result


def main():
    output = {
        "development": evaluate("full_development_combined_pairs.parquet", development_truth),
        "validation": evaluate("full_validation_combined_pairs.parquet", validation_truth),
    }
    destination = ROOT / "analysis" / "compact_candidate_probe_results.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
