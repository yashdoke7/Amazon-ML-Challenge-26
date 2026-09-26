"""Time the unchanged submission pipeline on a representative test slice."""

import json
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import FEATURE_NAMES, pair_features
from infer import candidates, query_batches


def main():
    data = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
    con = duckdb.connect(str(ROOT / ".duckdb" / "test_index.duckdb"), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    model = joblib.load(ROOT / "code" / "business_entity_resolution" / "model.joblib")
    rows = next(query_batches(data / "test_source1.tsv", 1000, None, 1000, 0))
    started = time.perf_counter()
    frame = candidates(con, rows)
    retrieval = time.perf_counter() - started
    started = time.perf_counter()
    matrix = np.empty((len(frame), len(FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        matrix[i] = pair_features(row.q_name, row.t_name, row.q_address, row.t_address, row.target_source)
    features = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba(matrix)[:, 1]
    scoring = time.perf_counter() - started
    started = time.perf_counter()
    ids = {row["entity_id"]: [] for row in rows}
    predictions = {row["entity_id"]: [] for row in rows}
    for (s1, target), probability in zip(frame[["s1_id", "target_id"]].itertuples(index=False, name=None), probabilities):
        ids[s1].append(target)
        if probability >= 0.65:
            predictions[s1].append(target)
    for row in rows:
        s1 = row["entity_id"]
        sorted(ids[s1]); sorted(predictions[s1])
    output = time.perf_counter() - started
    result = {"queries": len(rows), "candidate_pairs": len(frame),
              "retrieval_seconds": round(retrieval, 3),
              "string_feature_seconds": round(features, 3),
              "lightgbm_seconds": round(scoring, 3),
              "grouping_sorting_seconds": round(output, 3),
              "total_seconds": round(retrieval + features + scoring + output, 3)}
    print(json.dumps(result, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
