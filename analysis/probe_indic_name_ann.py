"""Measure full-corpus name ANN recall on missing Indic-script development links.

Builds a local HNSW index of *all* distinct Indic-script target names from the
supplied training sources, then queries Latin-script Source 1 names. The probe
only measures retrieval; it does not add links to a submission or train on
held-out labels. The Apache-2.0 multilingual MiniLM weights are cached locally.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from translit_miss_headroom import has_indic  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_candidates():
    with (DEV / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row["candidate_entity_ids"].split(","))
                if row["candidate_entity_ids"] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=["business_name", "business_address"],
                        default="business_name")
    args = parser.parse_args()
    field = args.field
    started = time.perf_counter()
    candidates = read_candidates()
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in candidates and row["country"] == "India":
                queries[row["entity_id"]] = row[field]
    missed = []
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in queries:
                missed.extend((q, t) for t in row["matched_entity_ids"].split(",")
                              if t and t not in candidates[q])
    missed_target_ids = {t for _, t in missed}
    target_name_index = {}
    name_to_index = {}
    names = []
    ids_per_name = []
    indic_records = 0
    for source in (2, 3):
        with (DATA / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["country"] != "India" or not row[field] or (
                        field == "business_name" and not has_indic(row["business_name"])):
                    continue
                indic_records += 1
                name = row[field]
                index = name_to_index.get(name)
                if index is None:
                    index = len(names)
                    name_to_index[name] = index
                    names.append(name)
                    ids_per_name.append(0)
                ids_per_name[index] += 1
                if row["entity_id"] in missed_target_ids:
                    target_name_index[row["entity_id"]] = index
        print("scanned source", source, "indic_records", indic_records,
              "distinct_names", len(names), "seconds", round(time.perf_counter()-started, 1), flush=True)
    misses = [(q, target_name_index[t]) for q, t in missed if t in target_name_index]
    assert len(misses) > 1000
    print("field", field, "eligible_misses", len(misses), "loading encoder", flush=True)
    torch.set_num_threads(4)
    encoder = SentenceTransformer(MODEL, device="cuda")
    vectors = np.empty((len(names), 384), dtype=np.float16)
    for first in range(0, len(names), 25_000):
        last = min(first+25_000, len(names))
        vectors[first:last] = encoder.encode(names[first:last], batch_size=128, device="cuda",
                                             show_progress_bar=False, normalize_embeddings=True)
        print("encoded", last, "of", len(names), "seconds", round(time.perf_counter()-started, 1), flush=True)
    print("building HNSW", len(names), flush=True)
    index = hnswlib.Index(space="cosine", dim=384)
    index.init_index(max_elements=len(names), ef_construction=80, M=16)
    index.set_num_threads(4)
    for first in range(0, len(names), 50_000):
        last = min(first+50_000, len(names))
        index.add_items(vectors[first:last], np.arange(first, last), num_threads=4)
        print("indexed", last, "of", len(names), "seconds", round(time.perf_counter()-started, 1), flush=True)
    index.set_ef(200)
    unique_queries = list(dict.fromkeys(q for q, _ in misses))
    query_vectors = encoder.encode([queries[q] for q in unique_queries], batch_size=128,
                                   device="cuda", show_progress_bar=False,
                                   normalize_embeddings=True)
    neighbors, distances = index.knn_query(query_vectors, k=100, num_threads=4)
    locations = {q: i for i, q in enumerate(unique_queries)}
    counts = defaultdict(int)
    for q, true_index in misses:
        i = locations[q]
        hits = neighbors[i]
        position = np.flatnonzero(hits == true_index)
        for k in (1, 5, 10, 20, 50, 100):
            counts[f"name_recall_at_{k}"] += bool(len(position) and position[0] < k)
    volumes = [sum(ids_per_name[j] for j in row[:50]) for row in neighbors]
    result = {"field": field,
              "all_target_records": indic_records,
              "all_target_distinct_texts": len(names),
              "eligible_missing_true_links": len(misses),
              "distinct_missed_queries": len(unique_queries),
              **dict(counts),
              "top50_id_volume_median": float(np.median(volumes)),
              "top50_id_volume_p90": float(np.percentile(volumes, 90)),
              "top50_id_volume_max": int(max(volumes)),
              "seconds": round(time.perf_counter()-started, 1)}
    destination = ROOT / "analysis" / (
        "indic_name_ann_probe_results.json" if field == "business_name"
        else "india_address_ann_probe_results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
