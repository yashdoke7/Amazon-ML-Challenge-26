"""Exact same-country full-index character TF-IDF rank for candidate misses.

Counts name-key rank only; duplicate target IDs per name make this an optimistic
retrieval ceiling rather than a production candidate or F0.5 score. It uses
only supplied records and locally computed sparse features.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import core  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"
QUOTAS = (1, 5, 10, 20, 50, 100)


def ids(path, field, quota=None):
    with path.open(encoding="utf-8", newline="") as stream:
        return {r["source1_entity_id"]: set(r[field].split(",")[:quota]) if r[field] else set()
                for r in csv.DictReader(stream, delimiter="\t")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", choices=("US", "India"), default="US")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    base = ids(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    address = ids(ROOT / "analysis" / "india_address_tfidf_dev_candidates.tsv",
                  "candidate_entity_ids", 20) if args.country == "India" else {}
    wanted_q = set(base)
    query_names = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in wanted_q and row["country"] == args.country:
                query_names[row["entity_id"]] = core(anyascii(row["business_name"]))
    misses = []
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in query_names and row["matched_entity_ids"]:
                pool = base[q] | address.get(q, set())
                misses.extend((q, t) for t in row["matched_entity_ids"].split(",")
                              if t not in pool)
    assert len(misses) == (2438 if args.country == "US" else 1953)
    wanted_target = {t for _, t in misses}
    names, name_index, volume, target_index = [], {}, [], {}
    record_count = 0
    for source in (2, 3):
        with (DATA / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["country"] != args.country:
                    continue
                record_count += 1
                raw = row["business_name"]
                index = name_index.get(raw)
                if index is None:
                    index = len(names)
                    name_index[raw] = index
                    names.append(core(anyascii(raw)))
                    volume.append(0)
                volume[index] += 1
                if row["entity_id"] in wanted_target:
                    target_index[row["entity_id"]] = index
    assert len(target_index) == len(wanted_target)
    print("targets", record_count, "name_keys", len(names),
          "missed_links", len(misses), "seconds", round(time.perf_counter()-started, 1),
          flush=True)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=2,
                                 max_df=0.8, max_features=120_000,
                                 sublinear_tf=True, dtype=np.float32)
    corpus = vectorizer.fit_transform(names)
    print("matrix", corpus.shape, "nnz", corpus.nnz,
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    del names, name_index
    qids = list(dict.fromkeys(q for q, _ in misses))
    wanted = {}
    for q, t in misses:
        wanted.setdefault(q, []).append(target_index[t])
    counts = Counter()
    for first in range(0, len(qids), args.batch_size):
        batch = qids[first:first+args.batch_size]
        encoded = vectorizer.transform([query_names[q] for q in batch])
        result = sp_matmul_topn(encoded, corpus.T, top_n=100,
                                n_threads=args.threads, sort=True)
        for j, q in enumerate(batch):
            lo, hi = result.indptr[j:j+2]
            positions = {int(idx): pos for pos, idx in enumerate(result.indices[lo:hi])}
            counts["queried"] += 1
            for index in wanted[q]:
                rank = positions.get(index, 1_000_000)
                counts["missing_links"] += 1
                for quota in QUOTAS:
                    counts[f"key_top{quota}"] += rank < quota
                if rank < 100:
                    counts["ranked_volume"] += volume[index]
        if first and first % 400 == 0:
            print("queried", min(first+args.batch_size, len(qids)),
                  "seconds", round(time.perf_counter()-started, 1), flush=True)
    result = {"country": args.country, "target_records": record_count,
              "unique_name_keys": len(volume), "missed_links": len(misses),
              "distinct_missed_queries": len(qids), "tfidf_features": corpus.shape[1],
              "tfidf_nnz": int(corpus.nnz), "counts": dict(counts),
              "seconds": round(time.perf_counter()-started, 1),
              "note": "Top-K is name-key rank; duplicate IDs and classifier may reduce actual link gain."}
    dest = ROOT / "analysis" / f"name_tfidf_{args.country.lower()}_missing_results.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
