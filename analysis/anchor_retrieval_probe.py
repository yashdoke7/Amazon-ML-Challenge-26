"""Measure whether final predicted target links can retrieve additional siblings.

Uses predicted anchors for retrieval. Ground truth is read only for evaluation.
The original candidate pool is retained; this script does not train or submit.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from infer import candidates  # noqa: E402
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def read(path, field, quota=None):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")[:quota])
                if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--split", choices=("development", "validation"),
                        default="development")
    parser.add_argument("--pairs-out", type=Path)
    parser.add_argument("--query-ids-out", type=Path)
    parser.add_argument("--result", type=Path,
                        default=ROOT / "analysis/anchor_retrieval_probe_results.json")
    args = parser.parse_args()
    if args.pairs_out and not args.rank:
        parser.error("--pairs-out requires --rank")
    started = time.perf_counter()
    if args.split == "development":
        base = read(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
        address = read(ROOT / "analysis/india_address_tfidf_dev_candidates.tsv",
                       "candidate_entity_ids", 20)
        chosen = read(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    else:
        base = read(DEV / "validation_tfidf_merged_candidates.tsv",
                    "candidate_entity_ids")
        address = {}
        chosen = read(DEV / "validation_tfidf_merged_final.tsv", "matched_entity_ids")
    truth = read(DATA / "train_ground_truth.tsv", "matched_entity_ids")
    rng = random.Random(20260927)
    qids = rng.sample(sorted(base), min(args.queries, len(base)))
    if args.query_ids_out:
        args.query_ids_out.parent.mkdir(parents=True, exist_ok=True)
        args.query_ids_out.write_text("\n".join(qids) + "\n", encoding="utf-8")
    anchors = set().union(*(chosen[q] for q in qids))
    print("sampled_queries", len(qids), "predicted_anchor_ids", len(anchors), flush=True)
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.execute("SET memory_limit='14GB'")
    con.execute("SET threads=8")
    con.register("wanted_anchors", pd.DataFrame({"target_id": list(anchors)}))
    frame = con.execute("""SELECT t.target_id entity_id,t.business_name,
        coalesce(t.business_address,'') business_address,t.country,t.source
        FROM wanted_anchors w JOIN target t USING(target_id)""").df()
    con.unregister("wanted_anchors")
    if len(frame) != len(anchors):
        raise ValueError("Not every predicted anchor exists in the target table")
    rows = frame.to_dict("records")
    sources = {row["entity_id"]: row["source"] for row in rows}
    recovered = defaultdict(set)
    recovered_cross_source = defaultdict(set)
    ranked = defaultdict(list)
    model = (joblib.load(ROOT / "code/business_entity_resolution/generalized_model.joblib")
             if args.rank else None)
    pair_count = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        result = candidates(con, batch, combined=True)
        pair_count += len(result)
        if model is not None and len(result):
            X = np.empty((len(result), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
            for i, row in enumerate(result.itertuples(index=False)):
                X[i] = generalized_pair_features(
                    row.q_name, row.t_name, row.q_address, row.t_address,
                    row.target_source)
            result["anchor_probability"] = model.predict_proba(X)[:, 1]
        for row in result.itertuples(index=False):
            if row.target_id == row.s1_id:
                continue
            recovered[row.s1_id].add(row.target_id)
            if row.target_source != sources[row.s1_id]:
                recovered_cross_source[row.s1_id].add(row.target_id)
            if model is not None:
                ranked[row.s1_id].append((row.target_id, row.anchor_probability))
        if start and start % 1000 < args.batch_size:
            print("anchors_done", min(start + args.batch_size, len(rows)),
                  "candidate_pairs", pair_count,
                  "seconds", round(time.perf_counter() - started, 1), flush=True)
    con.close()
    counts = Counter()
    ranked_ids = {}
    if model is not None:
        for anchor, pairs in ranked.items():
            ranked_ids[anchor] = sorted(
                pairs, key=lambda pair: (-pair[1], pair[0]))[:20]
    detailed = []
    for q in qids:
        pool = base[q] | address.get(q, set())
        missing = truth[q] - pool
        union = set().union(*(recovered[a] for a in chosen[q])) if chosen[q] else set()
        cross = (set().union(*(recovered_cross_source[a] for a in chosen[q]))
                 if chosen[q] else set())
        counts["missing_true_links"] += len(missing)
        counts["all_anchor_new_candidate_pairs"] += len(union - pool)
        counts["all_anchor_recovered_true"] += len(missing & union)
        counts["cross_source_new_candidate_pairs"] += len(cross - pool)
        counts["cross_source_recovered_true"] += len(missing & cross)
        if model is not None:
            for quota in (1, 3, 5, 10, 20):
                top = {t for a in chosen[q] for t, _ in ranked_ids.get(a, [])[:quota]}
                counts[f"rank_top{quota}_new_candidate_pairs"] += len(top - pool)
                counts[f"rank_top{quota}_recovered_true"] += len(missing & top)
            if args.pairs_out:
                novel = {}
                for anchor in chosen[q]:
                    for rank, (target, probability) in enumerate(
                            ranked_ids.get(anchor, [])[:5], 1):
                        if target not in pool and (target not in novel or
                                                   probability > novel[target][1]):
                            novel[target] = (anchor, probability, rank)
                detailed.extend((q, target, anchor, probability, rank)
                                for target, (anchor, probability, rank) in novel.items())
    if args.pairs_out:
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pairs_out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "target_id", "anchor_id",
                             "anchor_probability", "anchor_rank"])
            writer.writerows(detailed)
    result = {"sampled_queries": len(qids), "predicted_anchor_ids": len(anchors),
              "anchor_pair_count": pair_count, "counts": dict(counts),
              "seconds": round(time.perf_counter() - started, 1),
              "note": "Oracle candidate recall only; must test ranking, final F0.5 and size."}
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
