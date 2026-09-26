"""Stream a larger supplied-data sample and train with model-mined negatives.

The current 35-feature model mines hard nonmatches; the new model is trained on
fresh labels from sampled training-owned Source 1 groups. No test labels, remote
services, or external business data are used. Run only when full inference is
idle, because candidate features and DuckDB need several CPU cores and RAM.
"""

import csv
import gc
import json
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generalized_features import GENERALIZED_FEATURE_NAMES  # noqa: E402
from infer import candidates  # noqa: E402
from rescore_candidates import featurize_rows  # noqa: E402
from train import sample_truth  # noqa: E402
from validation import entity_split  # noqa: E402

SAMPLE = 200_000
BATCH = 500
HARD_NEGATIVES = 20
RANDOM_NEGATIVES = 5


def feature_matrix(pool, frame):
    rows = [(row.q_name, row.t_name, row.q_address, row.t_address,
             row.target_source) for row in frame.itertuples(index=False)]
    chunks = (rows[i:i+25_000] for i in range(0, len(rows), 25_000))
    return np.concatenate(list(pool.map(
        partial(featurize_rows, model_feature_names=GENERALIZED_FEATURE_NAMES), chunks)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--base-model", type=Path,
                        default=PACKAGE / "generalized_seed_model.joblib")
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=SAMPLE)
    args = parser.parse_args()
    started = time.perf_counter()
    truth, heldout_targets = sample_truth(args.data_dir, count=args.sample_size)
    truth = {q: ids for q, ids in truth.items() if entity_split(q) == "training"}
    queries = []
    with (args.data_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in truth:
                queries.append(row)
    assert len(queries) == len(truth)
    print("sampled training queries", len(queries), "heldout targets", len(heldout_targets),
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    con = duckdb.connect(str(args.db), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    base = joblib.load(args.base_model)
    rng = np.random.default_rng(20260926)
    pieces = []
    labels = []
    calibration = []
    counts = {"retrieved_pairs": 0, "retrieved_positives": 0,
              "kept_positives": 0, "kept_negatives": 0}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for first in range(0, len(queries), BATCH):
            batch = queries[first:first+BATCH]
            frame = candidates(con, batch, combined=False).fillna("")
            if frame.empty:
                continue
            matrix = feature_matrix(pool, frame)
            probs = base.predict_proba(matrix)[:, 1]
            s1 = frame.s1_id.to_numpy()
            targets = frame.target_id.to_numpy()
            positive = np.fromiter((target in truth[q] for q, target in zip(s1, targets)),
                                   dtype=bool, count=len(frame))
            allowed = positive | np.fromiter((target not in heldout_targets for target in targets),
                                              dtype=bool, count=len(frame))
            keep = []
            for query, indices in frame.groupby("s1_id", sort=False).indices.items():
                pos = indices[positive[indices]]
                neg = indices[(~positive[indices]) & allowed[indices]]
                keep.extend(pos)
                if len(neg):
                    ordered = neg[np.argsort(-probs[neg], kind="stable")]
                    hard = ordered[:HARD_NEGATIVES]
                    keep.extend(hard)
                    remaining = ordered[HARD_NEGATIVES:]
                    if len(remaining):
                        keep.extend(rng.choice(remaining,
                                               size=min(RANDOM_NEGATIVES, len(remaining)),
                                               replace=False))
            selected = np.asarray(keep, dtype=np.int64)
            pieces.append(matrix[selected].copy())
            labels.append(positive[selected].astype(np.int8))
            calibration.append(np.fromiter(
                (int(q.split("-")[-1]) % 10 == 0 for q in s1[selected]),
                dtype=bool, count=len(selected)))
            counts["retrieved_pairs"] += len(frame)
            counts["retrieved_positives"] += int(positive.sum())
            counts["kept_positives"] += int(positive[selected].sum())
            counts["kept_negatives"] += int((~positive[selected]).sum())
            if (first // BATCH) % 10 == 0:
                print("queries", first+len(batch), "retrieved_pairs", counts["retrieved_pairs"],
                      "kept", counts["kept_positives"]+counts["kept_negatives"],
                      "seconds", round(time.perf_counter()-started, 1), flush=True)
            del frame, matrix, probs, positive, allowed
            gc.collect()
    con.close()
    X = np.concatenate(pieces)
    y = np.concatenate(labels)
    cal = np.concatenate(calibration)
    assert len(X) == len(y) == len(cal)
    del pieces, labels, calibration
    gc.collect()
    print("fitting", len(X), "rows", int(y.sum()), "positives",
          "calibration", int(cal.sum()), "seconds",
          round(time.perf_counter()-started, 1), flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.04, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=8, verbosity=-1, random_state=20260926)
    model.fit(X[~cal], y[~cal], feature_name=GENERALIZED_FEATURE_NAMES,
              eval_set=[(X[cal], y[cal])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    result = {"sample_requested": args.sample_size, "queries": len(queries),
              **counts, "training_rows": len(X),
              "best_iteration": int(model.best_iteration_),
              "seconds": round(time.perf_counter()-started, 1)}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
