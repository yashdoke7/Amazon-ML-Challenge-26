"""Probe whether predicted sibling records help recover rejected candidate links.

This is a feasibility experiment on the older fixed 2k development/validation
candidate pools. It never uses truth to choose an anchor. A promising result
would still require training-owned reconstruction on the final candidate pool.
"""

import json
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from generalized_features import GENERALIZED_FEATURE_NAMES  # noqa: E402
from rescore_candidates import featurize_rows  # noqa: E402
from validation import f05_for_query  # noqa: E402
from calibrate_on_development import sampled_truth as dev_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as val_truth  # noqa: E402


def prepare(filename, truth_fn, matcher):
    start = time.perf_counter()
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("").reset_index(drop=True)
    rows = [(r.q_name, r.t_name, r.q_address, r.t_address, r.target_source)
            for r in frame.itertuples(index=False)]
    chunks = (rows[i:i+25_000] for i in range(0, len(rows), 25_000))
    with ProcessPoolExecutor(max_workers=4) as pool:
        X = np.concatenate(list(pool.map(
            partial(featurize_rows, model_feature_names=GENERALIZED_FEATURE_NAMES), chunks)))
    p = matcher.predict_proba(X)[:, 1]
    country = frame.country.to_numpy()
    base = p >= np.where(country == "India", 0.65, 0.75)
    anchors = defaultdict(list)
    for i in np.flatnonzero(base):
        r = frame.iloc[i]
        anchors[r.s1_id].append((r.target_id, r.t_name, r.t_address, r.target_source))
    viable = np.flatnonzero((~base) & (p >= 0.02) & frame.s1_id.isin(anchors).to_numpy())
    context = np.zeros((len(viable), 17), dtype=np.float32)
    for j, i in enumerate(viable):
        r = frame.iloc[i]
        siblings = [(n, a, s) for t, n, a, s in anchors[r.s1_id] if t != r.target_id]
        if not siblings:
            continue
        name = [fuzz.ratio(r.t_name, n) for n, _, _ in siblings]
        name_set = [fuzz.token_set_ratio(r.t_name, n) for n, _, _ in siblings]
        address = [fuzz.ratio(r.t_address, a) for _, a, _ in siblings]
        address_set = [fuzz.token_set_ratio(r.t_address, a) for _, a, _ in siblings]
        ascii_name = [fuzz.ratio(anyascii(r.t_name), anyascii(n)) for n, _, _ in siblings]
        cross = [k for k, (_, _, s) in enumerate(siblings) if s != r.target_source]
        context[j] = [
            p[i], min(len(siblings), 11), int(r.country == "India"),
            int(r.target_source == "S3"), max(name), max(name_set),
            max(address), max(address_set), max(ascii_name),
            max(min(n, a) for n, a in zip(name_set, address_set)),
            max((name_set[k] for k in cross), default=0),
            max((address_set[k] for k in cross), default=0),
            X[i, 0], X[i, 2], X[i, 8], X[i, 10], X[i, 15]]
    truth = truth_fn()
    labels = frame.is_match.to_numpy(dtype=np.int8)[viable]
    print(filename, "pairs", len(frame), "anchors", int(base.sum()),
          "viable", len(viable), "viable positives", int(labels.sum()),
          "seconds", round(time.perf_counter()-start, 1), flush=True)
    return frame, truth, base, viable, context, labels


def metric(data, extra, wanted=None):
    frame, truth, base, viable, _, _ = data
    selected = base.copy()
    selected[viable] |= extra
    predictions = defaultdict(set)
    for q, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(
            index=False, name=None):
        predictions[q].add(target)
    wanted = wanted or set(truth)
    values = [f05_for_query(truth[q], predictions[q]) for q in wanted]
    chosen = np.flatnonzero(selected)
    labels = frame.is_match.to_numpy(dtype=bool)
    return {"macro_f05": float(np.mean(values)),
            "selected": int(selected.sum()), "tp": int(labels[chosen].sum()),
            "fp": int(len(chosen)-labels[chosen].sum()),
            "added": int(extra.sum()), "added_tp": int(data[5][extra].sum())}


def fit_and_test(dev, val, columns):
    q = dev[0].s1_id.to_numpy()[dev[3]]
    fit = np.fromiter((zlib.crc32(s.encode()) % 2 == 0 for s in q), dtype=bool,
                      count=len(q))
    model = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.04,
        num_leaves=7, min_child_samples=50, reg_lambda=4.0,
        n_jobs=4, verbosity=-1, random_state=20260927)
    model.fit(dev[4][fit][:, columns], dev[5][fit])
    dev_p = model.predict_proba(dev[4][:, columns])[:, 1]
    val_p = model.predict_proba(val[4][:, columns])[:, 1]
    holdout_q = {s for s in dev[1] if zlib.crc32(s.encode()) % 2 != 0}
    sweep = {str(t): metric(dev, dev_p >= t, holdout_q) for t in
             (0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99)}
    chosen = max(sweep, key=lambda t: sweep[t]["macro_f05"])
    return {"training_rows": int(fit.sum()),
            "training_positives": int(dev[5][fit].sum()),
            "holdout_baseline": metric(dev, np.zeros(len(dev[3]), dtype=bool), holdout_q),
            "chosen_threshold": float(chosen), "holdout_chosen": sweep[chosen],
            "validation_baseline": metric(val, np.zeros(len(val[3]), dtype=bool)),
            "validation_chosen": metric(val, val_p >= float(chosen)),
            "threshold_sweep": sweep}


def main():
    matcher = joblib.load(ROOT / "code" / "business_entity_resolution" /
                          "generalized_model.joblib")
    dev = prepare("full_development_combined_pairs.parquet", dev_truth, matcher)
    val = prepare("full_validation_combined_pairs.parquet", val_truth, matcher)
    result = {
        "probability_only": fit_and_test(dev, val, [0, 1, 2, 3]),
        "probability_and_group_context": fit_and_test(dev, val, list(range(17)))}
    path = ROOT / "analysis" / "group_context_feasibility_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"threshold": v["chosen_threshold"],
                          "holdout": v["holdout_chosen"]["macro_f05"],
                          "validation": v["validation_chosen"]["macro_f05"]}
                      for k, v in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
