"""Train a 20k-query matcher with first-address-number features in isolation."""

import csv
import sys
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from infer import candidates
from number_features import NUMBER_FEATURE_NAMES, number_pair_features
from train import sample_truth
from validation import entity_split

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DB = ROOT / ".duckdb" / "train_index.duckdb"
OUTPUT = ROOT / "analysis" / "large_number_model.joblib"


def main():
    started = time.perf_counter()
    truth, heldout = sample_truth(DATA, count=20_000)
    queries = []
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in truth:
                queries.append(row)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=4")
    frames = []
    for first in range(0, len(queries), 1000):
        frame = candidates(con, queries[first:first+1000], combined=False)
        frames.append(frame)
        print("retrieved", min(first+1000, len(queries)), "queries", flush=True)
    con.close()
    df = pd.concat(frames, ignore_index=True)
    df["split"] = [entity_split(s) for s in df.s1_id]
    df["is_match"] = [t in truth[s] for s, t in zip(df.s1_id, df.target_id)]
    df = df.loc[~((df.split == "training") & ~df.is_match & df.target_id.isin(heldout))].reset_index(drop=True)
    print("training pool", len(df), "positives", int(df.is_match.sum()), flush=True)
    matrix = np.empty((len(df), len(NUMBER_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        matrix[i] = number_pair_features(row.q_name, row.t_name,
                                         row.q_address, row.t_address, row.target_source)
        if i and i % 250_000 == 0:
            print("features", i, "seconds", round(time.perf_counter()-started, 1), flush=True)
    labels = df.is_match.to_numpy(dtype=np.int8)
    internal = np.array([int(s.split("-")[-1]) % 10 == 0 for s in df.s1_id])
    fit = (df.split.to_numpy() == "training") & ~internal
    calibration = ((df.split.to_numpy() == "training") & internal) | (df.split.to_numpy() == "development")
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=4, verbosity=-1, random_state=20260925)
    model.fit(matrix[fit], labels[fit], feature_name=NUMBER_FEATURE_NAMES,
              eval_set=[(matrix[calibration], labels[calibration])],
              eval_metric="binary_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
    joblib.dump(model, OUTPUT)
    print("saved", OUTPUT, "best iteration", model.best_iteration_,
          "seconds", round(time.perf_counter()-started, 1), flush=True)


if __name__ == "__main__":
    main()
