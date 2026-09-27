"""Small supplied-data speed/overlap check for ING's Apache-2 sparse_dot_topn."""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import normalize  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"


def main():
    started = time.perf_counter()
    addresses, seen = [], set()
    for source in (2, 3):
        with (DATA / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["country"] == "US" and row["business_address"]:
                    raw = row["business_address"]
                    if raw not in seen:
                        seen.add(raw)
                        addresses.append(normalize(anyascii(raw)))
                        if len(addresses) >= 500_000:
                            break
        if len(addresses) >= 500_000:
            break
    queries = []
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["country"] == "US":
                queries.append(normalize(anyascii(row["business_address"])))
                if len(queries) >= 1000:
                    break
    loaded = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.8,
        max_features=200_000, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, dtype=np.float32)
    target = vectorizer.fit_transform(addresses)
    query = vectorizer.transform(queries)
    vectorized = time.perf_counter()
    regular = (query @ target.T).tocsr()
    scipy_done = time.perf_counter()
    topn = sp_matmul_topn(query, target.T, top_n=100, n_threads=8, sort=True)
    topn_done = time.perf_counter()
    same_top5 = 0
    for row in range(len(queries)):
        lo, hi = regular.indptr[row:row+2]
        values = regular.data[lo:hi]
        indices = regular.indices[lo:hi]
        k = min(100, len(values))
        best = np.argpartition(values, -k)[-k:] if k else np.empty(0, dtype=np.int64)
        best = best[np.argsort(-values[best], kind="stable")]
        expected = set(indices[best[:5]])
        lo, hi = topn.indptr[row:row+2]
        actual = set(topn.indices[lo:hi][:5])
        same_top5 += expected == actual
    result = {"targets": len(addresses), "queries": len(queries),
              "target_nnz": int(target.nnz), "regular_nnz": int(regular.nnz),
              "topn_nnz": int(topn.nnz),
              "seconds_load": round(loaded-started, 1),
              "seconds_vectorize": round(vectorized-loaded, 1),
              "seconds_scipy": round(scipy_done-vectorized, 3),
              "seconds_sparse_dot_topn": round(topn_done-scipy_done, 3),
              "same_top5_rows": same_top5}
    (ROOT / "analysis" / "sparse_topn_speed_probe_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
