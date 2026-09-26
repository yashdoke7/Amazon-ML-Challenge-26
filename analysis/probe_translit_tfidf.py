"""Exact full-index transliterated character-TFIDF retrieval feasibility.

The corpus is every distinct Indian-script target name from supplied training
sources. Development truth is used only after retrieval to measure rank.
"""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import normalize  # noqa: E402
from translit_miss_headroom import has_indic  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"


def main():
    started = time.perf_counter()
    with (DEV / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as stream:
        candidates = {row["source1_entity_id"]: set(row["candidate_entity_ids"].split(","))
                      if row["candidate_entity_ids"] else set()
                      for row in csv.DictReader(stream, delimiter="\t")}
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in candidates and row["country"] == "India":
                queries[row["entity_id"]] = row["business_name"]
    missed = []
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in queries:
                missed.extend((q, t) for t in row["matched_entity_ids"].split(",")
                              if t and t not in candidates[q])
    wanted = {t for _, t in missed}
    names = []
    name_index = {}
    id_volume = []
    target_name_index = {}
    count = 0
    for source in (2, 3):
        with (DATA / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                name = row["business_name"]
                if row["country"] != "India" or not has_indic(name):
                    continue
                count += 1
                index = name_index.get(name)
                if index is None:
                    index = len(names)
                    name_index[name] = index
                    names.append(normalize(anyascii(name)))
                    id_volume.append(0)
                id_volume[index] += 1
                if row["entity_id"] in wanted:
                    target_name_index[row["entity_id"]] = index
    misses = [(q, target_name_index[t]) for q, t in missed if t in target_name_index]
    assert len(misses) > 1000
    print("records", count, "names", len(names), "misses", len(misses),
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_df=0.8, sublinear_tf=True, dtype=np.float32)
    corpus = vectorizer.fit_transform(names)
    print("TFIDF", corpus.shape, "nonzeros", corpus.nnz,
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    query_ids = list(dict.fromkeys(q for q, _ in misses))
    encoded = vectorizer.transform([normalize(anyascii(queries[q])) for q in query_ids])
    positions = {q: i for i, q in enumerate(query_ids)}
    desired = [[] for _ in query_ids]
    for q, target_index in misses:
        desired[positions[q]].append(target_index)
    rank_counts = Counter()
    candidate_volumes = []
    for first in range(0, len(query_ids), 20):
        similarities = (encoded[first:first+20] @ corpus.T).tocsr()
        for j in range(similarities.shape[0]):
            i = first + j
            lo, hi = similarities.indptr[j:j+2]
            values = similarities.data[lo:hi]
            indices = similarities.indices[lo:hi]
            k = min(100, len(values))
            if k:
                best = np.argpartition(values, -k)[-k:]
                best = best[np.argsort(-values[best], kind="stable")]
                neighbors = indices[best]
            else:
                neighbors = np.empty(0, dtype=np.int64)
            top50 = neighbors[:50]
            candidate_volumes.append(sum(id_volume[int(n)] for n in top50))
            rank = {int(n): position for position, n in enumerate(neighbors)}
            for target_index in desired[i]:
                found = rank.get(target_index, 1_000_000)
                for quota in (1, 5, 10, 20, 50, 100):
                    rank_counts[f"name_recall_at_{quota}"] += found < quota
        if first % 200 == 0:
            print("queried", min(first+20, len(query_ids)), "of", len(query_ids),
                  "seconds", round(time.perf_counter()-started, 1), flush=True)
    result = {"all_target_indic_records": count,
              "all_target_distinct_indic_names": len(names),
              "indic_missing_true_links": len(misses),
              "distinct_missed_queries": len(query_ids),
              "tfidf_features": corpus.shape[1],
              **dict(rank_counts),
              "top50_id_volume_median": float(np.median(candidate_volumes)),
              "top50_id_volume_p90": float(np.percentile(candidate_volumes, 90)),
              "top50_id_volume_max": int(max(candidate_volumes)),
              "seconds": round(time.perf_counter()-started, 1)}
    destination = ROOT / "analysis" / "translit_tfidf_probe_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
