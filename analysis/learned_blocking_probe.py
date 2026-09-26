"""Test a size-bounded learned block followed by the 27-feature matcher.

The broad lexical route is a first retrieval stage. A separate frozen 23-feature
model ranks those candidates; only the highest K per query are passed to the
final 27-feature matcher and written as candidate_pairs.tsv.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from number_features import NUMBER_FEATURE_NAMES, number_pair_features  # noqa: E402
from validation import f05_for_query  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

DATA = ROOT / "analysis"
PACKAGE = ROOT / "code" / "business_entity_resolution"
OPTIONS = [30, 35, 40, 45, 50]


def metric(frame, mask, truth):
    groups = defaultdict(set)
    for query, target in frame.loc[mask, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        groups[query].add(target)
    return {"macro_f05": round(float(np.mean([
        f05_for_query(ids, groups[q]) for q, ids in truth.items()])), 6),
        "tp": sum(len(groups[q] & ids) for q, ids in truth.items()),
        "fp": sum(len(groups[q] - ids) for q, ids in truth.items()),
        "selected": int(mask.sum())}


def evaluate(filename, truth_fn, blocker, matcher):
    frame = pd.read_parquet(DATA / filename).fillna("").reset_index(drop=True)
    truth = truth_fn()
    features = np.empty((len(frame), len(NUMBER_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        features[i] = number_pair_features(row.q_name, row.t_name,
                                            row.q_address, row.t_address,
                                            row.target_source)
    blocker_probs = blocker.predict_proba(features[:, :23])[:, 1]
    matcher_probs = matcher.predict_proba(features)[:, 1]
    final_mask = matcher_probs >= np.where(frame.country.to_numpy() == "India", 0.65, 0.75)
    frame["blocker_prob"] = blocker_probs
    frame["blocker_rank"] = frame.sort_values(
        ["s1_id", "blocker_prob", "target_id"],
        ascending=[True, False, True], kind="stable"
    ).groupby("s1_id", sort=False).cumcount().reindex(frame.index).to_numpy() + 1
    # The rank assignment above reindexes a Series in original row order.
    result = {"queries": len(truth), "input_pairs": len(frame),
              "full_matcher": metric(frame, final_mask, truth), "options": {}}
    truth_mask = frame.is_match.to_numpy(dtype=bool)
    for cap in OPTIONS:
        candidate_mask = frame.blocker_rank.to_numpy() <= cap
        candidate_truth = candidate_mask & truth_mask
        filtered_match = candidate_mask & final_mask
        result["options"][f"top{cap}"] = {
            "candidate_pairs": int(candidate_mask.sum()),
            "pairs_per_query": round(float(candidate_mask.sum() / len(truth)), 2),
            "positive_links": int(candidate_truth.sum()),
            "edge_recall": round(float(candidate_truth.sum() /
                                       sum(map(len, truth.values()))), 6),
            "oracle": metric(frame, candidate_truth, truth)["macro_f05"],
            "matcher": metric(frame, filtered_match, truth),
            "estimated_test_zip_mb_at_5p7_bytes_per_pair": round(
                (candidate_mask.sum() / len(truth)) * 1_732_544 * 5.7 / 1e6, 1),
        }
    return result


def main():
    blocker = joblib.load(PACKAGE / "model.joblib")
    matcher = joblib.load(PACKAGE / "number_model.joblib")
    result = {
        "development": evaluate("full_development_combined_pairs.parquet", development_truth,
                                 blocker, matcher),
        "validation": evaluate("full_validation_combined_pairs.parquet", validation_truth,
                                blocker, matcher),
    }
    destination = DATA / "learned_blocking_probe_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
