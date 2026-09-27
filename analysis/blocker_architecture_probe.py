"""Compare candidate pruning strategies on identical fixed broad candidate pools.

The production top-40 stage ranks broad pairs with the 23-feature blocker even
though it has already computed all 35 matcher features. This probes whether the
matcher itself or a hybrid order retains more labeled links at the same quota.
It leaves full-test output untouched and evaluates the two frozen split samples.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from features import FEATURE_NAMES  # noqa: E402
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from evaluate_generalized_model import score  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402
from validation import f05_for_query  # noqa: E402


def top_k(frame, probabilities, k):
    rank = frame[["s1_id", "target_id"]].copy()
    rank["prob"] = probabilities
    ordered = rank.sort_values(["s1_id", "prob", "target_id"],
                               ascending=[True, False, True], kind="stable")
    winners = ordered.groupby("s1_id", sort=False).head(k).index
    keep = np.zeros(len(frame), dtype=bool)
    keep[winners] = True
    return keep


def evaluate(frame, truth, final_p, keep):
    pool = frame.loc[keep].reset_index(drop=True)
    probabilities = final_p[keep]
    labels = pool.is_match.to_numpy(dtype=bool)
    reachable = int(labels.sum())
    oracle = {}
    for q, t in pool.loc[labels, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        oracle.setdefault(q, set()).add(t)
    oracle_macro = float(np.mean([f05_for_query(ids, oracle.get(q, set()))
                                  for q, ids in truth.items()]))
    metrics = score(pool, truth, probabilities, {"India": 0.65, "other": 0.75})
    return {"candidate_pairs": int(keep.sum()), "reachable_true_links": reachable,
            "oracle_macro_f05": oracle_macro, "decisions": metrics}


def run(filename, truth, blocker, matcher, started):
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    X = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        X[i] = generalized_pair_features(row.q_name, row.t_name, row.q_address,
                                         row.t_address, row.target_source)
        if i and i % 200_000 == 0:
            print(filename, "features", i, "seconds",
                  round(time.perf_counter()-started, 1), flush=True)
    blocker_p = blocker.predict_proba(X[:, :len(FEATURE_NAMES)])[:, 1]
    final_p = matcher.predict_proba(X)[:, 1]
    keep_blocker = top_k(frame, blocker_p, 40)
    keep_matcher = top_k(frame, final_p, 40)
    keep_hybrid = top_k(frame, blocker_p, 20) | top_k(frame, final_p, 20)
    result = {"queries": len(truth), "broad_pairs": len(frame),
              "broad_reachable": int(frame.is_match.sum()),
              "blocker_top40": evaluate(frame, truth, final_p, keep_blocker),
              "matcher_top40": evaluate(frame, truth, final_p, keep_matcher),
              "hybrid_top20_each": evaluate(frame, truth, final_p, keep_hybrid),
              "broad_unpruned": evaluate(frame, truth, final_p,
                                         np.ones(len(frame), dtype=bool))}
    print(filename, {k: (v["reachable_true_links"],
                          round(v["decisions"]["macro_f05"], 6))
                     for k, v in result.items() if isinstance(v, dict)}, flush=True)
    return result


def main():
    started = time.perf_counter()
    blocker = joblib.load(ROOT / "code" / "business_entity_resolution" / "model.joblib")
    matcher = joblib.load(ROOT / "code" / "business_entity_resolution" / "generalized_model.joblib")
    assert list(blocker.feature_name_) == FEATURE_NAMES
    assert list(matcher.feature_name_) == GENERALIZED_FEATURE_NAMES
    results = {"development": run("full_development_combined_pairs.parquet",
                                  development_truth(), blocker, matcher, started),
               "validation": run("full_validation_combined_pairs.parquet",
                                 validation_truth(), blocker, matcher, started)}
    results["seconds"] = round(time.perf_counter()-started, 1)
    (ROOT / "analysis" / "blocker_architecture_probe_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", results["seconds"], flush=True)


if __name__ == "__main__":
    main()
