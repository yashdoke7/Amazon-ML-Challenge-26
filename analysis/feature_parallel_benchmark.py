"""Compare exact feature extraction with multiple CPU processes on saved pairs."""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))
from features import FEATURE_NAMES, pair_features


def featurize(rows):
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(rows):
        matrix[index] = pair_features(row[0], row[2], row[1], row[3], row[4])
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("analysis/full_validation_combined_pairs.parquet"))
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    frame = pd.read_parquet(args.pairs, columns=["q_name", "q_address", "t_name", "t_address", "target_source"]).head(args.sample)
    rows = list(frame.itertuples(index=False, name=None))
    start = time.perf_counter()
    sequential = featurize(rows)
    sequential_seconds = time.perf_counter() - start
    chunk_size = (len(rows) + args.workers - 1) // args.workers
    chunks = [rows[offset:offset + chunk_size] for offset in range(0, len(rows), chunk_size)]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parallel = np.concatenate(list(pool.map(featurize, chunks)))
    parallel_seconds = time.perf_counter() - start
    print({"pairs": len(rows), "workers": args.workers,
           "sequential_seconds": round(sequential_seconds, 2),
           "parallel_seconds": round(parallel_seconds, 2),
           "speedup": round(sequential_seconds / parallel_seconds, 2),
           "bit_identical": bool(np.array_equal(sequential, parallel))}, flush=True)


if __name__ == "__main__":
    main()
