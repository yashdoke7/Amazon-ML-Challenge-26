"""Rebuild the 36-feature blank-target-address specialist from supplied TSVs."""

import argparse
import csv
import json
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features
from infer import candidates
from train import sample_truth
from validation import entity_split

FEATURE_NAMES = GENERALIZED_FEATURE_NAMES + ["log_target_core_frequency"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()
    started = time.perf_counter()
    sampled, heldout_targets = sample_truth(args.data_dir, count=args.sample_size)
    truth = {q: ids for q, ids in sampled.items() if entity_split(q) == "training"}
    queries = []
    with (args.data_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in truth:
                queries.append(row)
    con = duckdb.connect(str(args.db), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    pieces, labels, calibration = [], [], []
    total = positives = 0
    for first in range(0, len(queries), 500):
        frame = candidates(con, queries[first:first+500], combined=True).fillna("")
        frame = frame.loc[frame.t_address == ""].reset_index(drop=True)
        if frame.empty:
            continue
        frame["is_match"] = [t in truth[q] for q, t in zip(frame.s1_id, frame.target_id)]
        frame = frame.loc[frame.is_match | ~frame.target_id.isin(heldout_targets)].reset_index(drop=True)
        con.register("blank_batch", pd.DataFrame({"target_id": frame.target_id.drop_duplicates()}))
        df = dict(con.execute("""SELECT b.target_id,f.df FROM blank_batch b
            JOIN blank_target_frequency f USING(target_id)""").fetchall())
        con.unregister("blank_batch")
        assert len(df) == frame.target_id.nunique()
        features = np.empty((len(frame), len(FEATURE_NAMES)), dtype=np.float32)
        for i, row in enumerate(frame.itertuples(index=False)):
            features[i, :35] = generalized_pair_features(
                row.q_name, row.t_name, row.q_address, row.t_address,
                row.target_source)
            features[i, 35] = np.log1p(df[row.target_id])
        pieces.append(features)
        labels.append(frame.is_match.to_numpy(dtype=np.int8))
        calibration.append(np.fromiter(
            (int(q.split("-")[-1]) % 10 == 0 for q in frame.s1_id),
            dtype=bool, count=len(frame)))
        total += len(frame)
        positives += int(frame.is_match.sum())
        print("queries", min(first+500, len(queries)), "blank_pairs", total,
              "positives", positives, "seconds", round(time.perf_counter()-started, 1), flush=True)
    con.close()
    X = np.concatenate(pieces)
    y = np.concatenate(labels)
    cal = np.concatenate(calibration)
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=4, verbosity=-1, random_state=20260926)
    model.fit(X[~cal], y[~cal], feature_name=FEATURE_NAMES,
              eval_set=[(X[cal], y[cal])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(30, verbose=False)])
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    result = {"sample_requested": args.sample_size, "training_queries": len(queries),
              "blank_pairs": total, "positives": positives,
              "best_iteration": int(model.best_iteration_),
              "seconds": round(time.perf_counter()-started, 1)}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
