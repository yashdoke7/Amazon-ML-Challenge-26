"""Inspect deceptive exact-name/address candidates on a uniform S1 sample."""
import csv
import heapq
import json
import random
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
OUT = Path(__file__).resolve().parent / "collision_eda_results.json"
RNG = random.Random(20260925)
N = 5000
sample = []
with (ROOT / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
        item = (row["source1_entity_id"], set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set())
        if i < N:
            sample.append(item)
        else:
            j = RNG.randrange(i + 1)
            if j < N:
                sample[j] = item
truth = dict(sample)
queries = []
with (ROOT / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in truth:
            queries.append(row)

con = duckdb.connect()
con.execute("SET memory_limit='10GB'")
con.register("queries", pd.DataFrame(queries))
targets = " UNION ALL ".join(
    f"SELECT entity_id, business_name, business_address, country FROM read_csv('{(ROOT / f'train_source{s}.tsv').as_posix()}', delim='\t', header=true)"
    for s in (2, 3)
)
result = {"sample_s1": len(queries), "sample_true_edges": sum(map(len, truth.values()))}
for route, field in (("exact_name", "business_name"), ("exact_address", "business_address")):
    print(f"joining {route}", flush=True)
    sql = f"""
    SELECT q.entity_id AS s1_id, q.business_name AS s1_name, q.business_address AS s1_address,
           q.country, t.entity_id AS target_id, t.business_name AS target_name, t.business_address AS target_address
    FROM queries q JOIN ({targets}) t
      ON lower(q.{field}) = lower(t.{field}) AND q.country = t.country
    WHERE q.{field} != ''
    """
    rows = con.execute(sql).fetchall()
    counts = Counter()
    per_query = Counter()
    score_bins = Counter()
    false_cases = []
    true_cases = []
    hardest_false = []
    weakest_true = []
    for s1_id, s1_name, s1_address, country, target_id, target_name, target_address in rows:
        is_true = target_id in truth[s1_id]
        counts["all_pairs"] += 1
        per_query[s1_id] += 1
        counts["true_pairs"] += is_true
        counts["false_pairs"] += not is_true
        counts[f"{country}_false"] += not is_true
        counts[f"{country}_true"] += is_true
        item = {"s1_id": s1_id, "s1_name": s1_name, "s1_address": s1_address,
                "target_id": target_id, "target_name": target_name, "target_address": target_address,
                "country": country, "other_field_similarity": round(fuzz.token_sort_ratio(
                    (s1_address or "").casefold() if route == "exact_name" else (s1_name or "").casefold(),
                    (target_address or "").casefold() if route == "exact_name" else (target_name or "").casefold()), 1)}
        score = item["other_field_similarity"]
        other_exact = (s1_address or "").casefold() == (target_address or "").casefold() if route == "exact_name" else (s1_name or "").casefold() == (target_name or "").casefold()
        counts[f"both_fields_exact_{'true' if is_true else 'false'}"] += other_exact
        score_bin = "missing" if route == "exact_name" and not target_address else "<30" if score < 30 else "30-49" if score < 50 else "50-69" if score < 70 else "70-89" if score < 90 else "90+"
        score_bins[f"{'true' if is_true else 'false'}:{score_bin}"] += 1
        if not is_true:
            heapq.heappush(hardest_false, (score, counts["all_pairs"], item))
            if len(hardest_false) > 20:
                heapq.heappop(hardest_false)
        else:
            heapq.heappush(weakest_true, (-score, counts["all_pairs"], item))
            if len(weakest_true) > 20:
                heapq.heappop(weakest_true)
        reservoir = true_cases if is_true else false_cases
        seen = counts["true_pairs"] if is_true else counts["false_pairs"]
        if len(reservoir) < 30:
            reservoir.append(item)
        else:
            j = RNG.randrange(seen)
            if j < 30:
                reservoir[j] = item
    volumes = np.array([per_query.get(q["entity_id"], 0) for q in queries])
    query_lookup = {q["entity_id"]: q for q in queries}
    largest_queries = [{"s1": query_lookup[s1], "candidate_count": count} for s1, count in per_query.most_common(10)]
    result[route] = {"counts": counts, "query_candidate_counts": {"queries_with_any": int(np.count_nonzero(volumes)),
                     "median": float(np.quantile(volumes, 0.5)), "p90": float(np.quantile(volumes, 0.9)),
                     "p99": float(np.quantile(volumes, 0.99)), "max": int(volumes.max())}, "largest_queries": largest_queries,
                     "other_field_similarity_bins": score_bins,
                     "false_examples": false_cases, "true_examples": true_cases,
                     "hardest_false_examples": [v[2] for v in sorted(hardest_false, reverse=True)],
                     "weakest_true_examples": [v[2] for v in sorted(weakest_true, reverse=True)]}
    print(route, dict(counts), flush=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
