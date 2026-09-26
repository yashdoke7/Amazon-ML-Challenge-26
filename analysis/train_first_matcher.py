"""Train a first local matcher on the saved 5k-query core-route candidate sample.

This is an exploratory baseline. The small validation slice is reported honestly;
it must be repeated on more entities and with the final candidate routes.
"""

import csv
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
import regex
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from validation import entity_split, f05_for_query  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
PAIRS = Path(os.environ.get("MATCHER_PAIRS", str(ROOT / "analysis" / "full_core_pairs.parquet")))
OUT = ROOT / "analysis" / "first_matcher_results.json"
MODEL_OUT = ROOT / "analysis" / "first_matcher_model.joblib"
TOKEN_RX = regex.compile(r"[\p{L}\p{M}\p{N}]+")
NUMBER_RX = re.compile(r"\d+")
LEGAL_RX = re.compile(r"\b(?:inc|llc|ltd|limited|private|pvt|corp|corporation|llp|co|company|sas|sarl|sa|eurl)\b")


@lru_cache(maxsize=1_000_000)
def normalize(value):
    return " ".join(TOKEN_RX.findall(unicodedata.normalize("NFKC", value or "").casefold()))


@lru_cache(maxsize=1_000_000)
def core(value):
    return " ".join(LEGAL_RX.sub(" ", normalize(value)).split())


@lru_cache(maxsize=1_000_000)
def fold(value):
    return "".join(c for c in unicodedata.normalize("NFKD", core(value)) if not unicodedata.combining(c))


@lru_cache(maxsize=1_000_000)
def parts(value):
    norm = normalize(value)
    return frozenset(x for x in norm.split() if len(x) >= 2)


@lru_cache(maxsize=1_000_000)
def numbers(value):
    return frozenset(x.lstrip("0") or "0" for x in NUMBER_RX.findall(value or ""))


def pair_features(q_name, t_name, q_addr, t_addr, source):
    qn, tn = normalize(q_name), normalize(t_name)
    qc, tc = core(q_name), core(t_name)
    qa, ta = normalize(q_addr), normalize(t_addr)
    qnt, tnt = parts(q_name), parts(t_name)
    qat, tat = parts(q_addr), parts(t_addr)
    qnums, tnums = numbers(q_addr), numbers(t_addr)
    name_common = len(qnt & tnt)
    addr_common = len(qat & tat)
    common_nums = len(qnums & tnums)
    return (
        fuzz.ratio(qn, tn), fuzz.token_sort_ratio(qn, tn), fuzz.token_set_ratio(qn, tn),
        fuzz.ratio(qc, tc), fuzz.token_sort_ratio(qc, tc), fuzz.ratio(fold(q_name), fold(t_name)),
        int(bool(qc) and qc == tc), int(bool(qn) and qn == tn),
        fuzz.ratio(qa, ta), fuzz.token_sort_ratio(qa, ta), fuzz.token_set_ratio(qa, ta),
        name_common, name_common / max(1, min(len(qnt), len(tnt))),
        addr_common, addr_common / max(1, min(len(qat), len(tat))),
        common_nums, int(bool(qnums and tnums) and common_nums == 0),
        int(bool(t_addr)), len(qn), len(tn), len(qa), len(ta),
        int(source == "S3"),
    )


FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "core_ratio", "core_token_sort",
    "accent_core_ratio", "core_exact", "name_exact", "address_ratio", "address_token_sort",
    "address_token_set", "name_common_tokens", "name_containment", "address_common_tokens",
    "address_containment", "common_address_numbers", "disjoint_address_numbers",
    "target_has_address", "query_name_len", "target_name_len", "query_address_len",
    "target_address_len", "source3",
]


def load_truth_and_heldout():
    rng = random.Random(20260925)
    sample = []
    heldout_targets = set()
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            s1 = row["source1_entity_id"]
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            if entity_split(s1) != "training":
                heldout_targets.update(ids)
            if i < 5000:
                sample.append((s1, ids))
            else:
                j = rng.randrange(i + 1)
                if j < 5000:
                    sample[j] = (s1, ids)
    return {s1: set(ids) for s1, ids in sample}, heldout_targets


def score_subset(ids, truth, frame, probabilities, threshold):
    predictions = defaultdict(set)
    selected = np.flatnonzero(probabilities >= threshold)
    for i in selected:
        predictions[frame.iloc[i]["s1_id"]].add(frame.iloc[i]["target_id"])
    scores = [f05_for_query(truth[s1], predictions[s1]) for s1 in ids]
    tp = sum(len(truth[s1] & predictions[s1]) for s1 in ids)
    fp = sum(len(predictions[s1] - truth[s1]) for s1 in ids)
    fn = sum(len(truth[s1] - predictions[s1]) for s1 in ids)
    singleton_ids = [s1 for s1 in ids if not truth[s1]]
    return {"macro_f05": float(np.mean(scores)), "tp": tp, "fp": fp, "fn": fn,
            "pair_precision": tp / max(1, tp + fp), "pair_recall": tp / max(1, tp + fn),
            "singleton_accuracy": sum(not predictions[s1] for s1 in singleton_ids) / max(1, len(singleton_ids)),
            "query_count": len(ids), "singleton_count": len(singleton_ids)}


def main():
    tic = time.perf_counter()
    if not PAIRS.exists():
        raise SystemExit(f"Missing local candidate extract: {PAIRS}")
    truth, heldout_targets = load_truth_and_heldout()
    df = pd.read_parquet(PAIRS)
    assert len(df) == len(df[["s1_id", "target_id"]].drop_duplicates())
    assert set(df.s1_id).issubset(truth)
    observed_labels = [target in truth[s1] for s1, target in zip(df.s1_id, df.target_id)]
    assert np.array_equal(df.is_match.to_numpy(), np.asarray(observed_labels))
    unsafe = df[(df.split == "training") & (~df.is_match) & df.target_id.isin(heldout_targets)]
    assert unsafe.empty, f"{len(unsafe)} training negatives belong to held-out entities"
    print("validated extract",len(df),"pairs and",int(df.is_match.sum()),"positives",flush=True)

    features = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        features[i] = pair_features(row.q_name, row.t_name, row.q_address,
                                    row.t_address, row.target_source)
        if i and i % 100_000 == 0:
            print("features",i,"seconds",round(time.perf_counter()-tic,1),flush=True)
    labels = df.is_match.to_numpy(dtype=np.int8)
    # An internal calibration fold inside the frozen training partition provides
    # enough entities for threshold selection while preserving validation.
    internal_cal = np.array([int(s1.split("-")[-1]) % 10 == 0 for s1 in df.s1_id])
    fit_mask = (df.split == "training").to_numpy() & ~internal_cal
    cal_mask = ((df.split == "training").to_numpy() & internal_cal) | (df.split == "development").to_numpy()
    val_mask = (df.split == "validation").to_numpy()
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=8, verbosity=-1, random_state=20260925)
    model.fit(features[fit_mask], labels[fit_mask], feature_name=FEATURE_NAMES,
              eval_set=[(features[cal_mask], labels[cal_mask])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(30,verbose=False)])
    cal_frame = df.loc[cal_mask].reset_index(drop=True)
    val_frame = df.loc[val_mask].reset_index(drop=True)
    cal_probs = model.predict_proba(features[cal_mask])[:,1]
    val_probs = model.predict_proba(features[val_mask])[:,1]
    cal_ids = sorted(s1 for s1 in truth if entity_split(s1) == "development" or
                     entity_split(s1) == "training" and int(s1.split("-")[-1]) % 10 == 0)
    val_ids = sorted(s1 for s1 in truth if entity_split(s1) == "validation")
    thresholds = list(np.linspace(0.05,0.95,19)) + [0.97,0.98,0.99,0.995]
    cal_scores = [(float(t),score_subset(cal_ids,truth,cal_frame,cal_probs,t)) for t in thresholds]
    best_t,best_cal = max(cal_scores,key=lambda item:item[1]["macro_f05"])
    val_score = score_subset(val_ids,truth,val_frame,val_probs,best_t)
    out = {"method":"LightGBM first core-candidate baseline", "candidate_file":PAIRS.name,
           "exploratory_sample_s1":len(truth),
           "candidate_rows":len(df),"candidate_true_rows":int(labels.sum()),
           "fit_rows":int(fit_mask.sum()),"calibration_rows":int(cal_mask.sum()),
           "validation_rows":int(val_mask.sum()),"best_iteration":model.best_iteration_,
           "calibration_threshold":best_t,"calibration_score":best_cal,
           "validation_score":val_score,"seconds":round(time.perf_counter()-tic,1),
           "feature_importance":dict(sorted(zip(FEATURE_NAMES,model.feature_importances_.tolist()),
                                        key=lambda item:item[1],reverse=True))}
    OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
    joblib.dump(model, MODEL_OUT)
    print(json.dumps({k:out[k] for k in ("candidate_rows","best_iteration","calibration_threshold",
        "calibration_score","validation_score","seconds")},indent=2),flush=True)
    print("wrote",OUT,"and",MODEL_OUT,flush=True)


if __name__ == "__main__":
    main()
