"""Create a compact learned block and final matches from broad saved candidates.

The 23-feature model is used only to rank the broad lexical candidate pool.
At most K candidates per Source 1 pass to the distinct 27-feature final matcher.
The written candidate TSV is precisely that last set passed to the matcher.
Both models use only locally supplied records and bundled frozen weights.
"""

import argparse
import csv
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES
from model_features import extractor_for
from rescore_candidates import featurize_rows


def process_batch(con, blocker, matcher, rows, candidate_writer, match_writer,
                  pool, top_k, thresholds, default_threshold, model_feature_names):
    queries = {q["entity_id"]: q for q, _ in rows}
    pairs = [(q["entity_id"], target) for q, ids in rows for target in ids]
    retained = {q["entity_id"]: [] for q, _ in rows}
    predictions = {q["entity_id"]: [] for q, _ in rows}
    if pairs:
        con.register("batch_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id"]))
        frame = con.execute("""
            SELECT p.s1_id,p.target_id,t.business_name,t.business_address,t.source
            FROM batch_pairs p JOIN target t USING(target_id)
        """).df()
        con.unregister("batch_pairs")
        if len(frame) != len(pairs):
            raise ValueError("Broad candidate target is missing from the target index")
        feature_rows = []
        for row in frame.itertuples(index=False):
            q = queries[row.s1_id]
            feature_rows.append((q["business_name"], row.business_name,
                                 q["business_address"], row.business_address,
                                 row.source))
        chunks = (feature_rows[i:i+50_000] for i in range(0, len(feature_rows), 50_000))
        features = np.concatenate(list(pool.map(
            partial(featurize_rows, model_feature_names=model_feature_names), chunks)))
        blocker_probs = blocker.predict_proba(features[:, :len(FEATURE_NAMES)])[:, 1]
        frame["blocker_prob"] = blocker_probs
        frame["feature_index"] = np.arange(len(frame))
        frame = frame.sort_values(["s1_id", "blocker_prob", "target_id"],
                                  ascending=[True, False, True], kind="stable")
        frame = frame.loc[frame.groupby("s1_id", sort=False).cumcount() < top_k].copy()
        # The retained rows are the exact last-stage candidates passed to the
        # final matcher and reported in the submission artifact.
        frame["final_prob"] = matcher.predict_proba(
            features[frame["feature_index"].to_numpy()])[:, 1]
        for row in frame.itertuples(index=False):
            retained[row.s1_id].append(row.target_id)
            threshold = thresholds.get(queries[row.s1_id]["country"], default_threshold)
            if row.final_prob >= threshold:
                predictions[row.s1_id].append(row.target_id)
    for q, _ in rows:
        s1 = q["entity_id"]
        candidate_writer.writerow([s1, ",".join(sorted(retained[s1]))])
        match_writer.writerow([s1, ",".join(sorted(predictions[s1]))])
    return len(pairs), sum(map(len, retained.values())), sum(map(len, predictions.values()))


def run(data_dir, db_path, broad_path, candidate_path, matching_path,
        top_k, workers, batch_size, maximum, thresholds, default_threshold,
        matcher_path, wanted):
    if len({broad_path.resolve(), candidate_path.resolve(), matching_path.resolve()}) != 3:
        raise ValueError("Input and output paths must be distinct")
    if top_k < 1 or workers < 1 or batch_size < 1:
        raise ValueError("top-k, workers, and batch-size must be positive")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    package = Path(__file__).resolve().parents[1]
    blocker = joblib.load(package / "model.joblib")
    matcher = joblib.load(matcher_path or package / "number_model.joblib")
    model_feature_names, _ = extractor_for(matcher)
    if list(blocker.feature_name_) != FEATURE_NAMES or model_feature_names[:len(FEATURE_NAMES)] != FEATURE_NAMES:
        raise ValueError("Unexpected bundled model feature contract")
    started = time.perf_counter()
    total_queries = broad_pairs = final_pairs = matches = 0
    query_path = data_dir / f"{data_dir.name}_source1.tsv"
    with ProcessPoolExecutor(max_workers=workers) as pool, \
         query_path.open(encoding="utf-8", newline="") as query_stream, \
         broad_path.open(encoding="utf-8", newline="") as broad_stream, \
         candidate_path.open("w", encoding="utf-8", newline="") as candidate_stream, \
         matching_path.open("w", encoding="utf-8", newline="") as match_stream:
        queries = csv.DictReader(query_stream, delimiter="\t")
        if wanted is not None:
            queries = (query for query in queries if query["entity_id"] in wanted)
        broad = csv.DictReader(broad_stream, delimiter="\t")
        if broad.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Unexpected broad candidate header")
        candidate_writer = csv.writer(candidate_stream, delimiter="\t", lineterminator="\n")
        match_writer = csv.writer(match_stream, delimiter="\t", lineterminator="\n")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        batch = []
        for q, c in itertools.zip_longest(queries, broad):
            if q is None or c is None:
                raise ValueError("Query and broad candidate row counts differ")
            if q["entity_id"] != c["source1_entity_id"]:
                raise ValueError("Query and broad candidate IDs differ")
            ids = c["candidate_entity_ids"].split(",") if c["candidate_entity_ids"] else []
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate broad candidate")
            batch.append((q, ids))
            if len(batch) >= batch_size:
                n_broad, n_final, n_match = process_batch(
                    con, blocker, matcher, batch, candidate_writer, match_writer,
                    pool, top_k, thresholds, default_threshold, model_feature_names)
                total_queries += len(batch)
                broad_pairs += n_broad
                final_pairs += n_final
                matches += n_match
                candidate_stream.flush()
                match_stream.flush()
                print("queries", total_queries, "broad_pairs", broad_pairs,
                      "final_candidates", final_pairs, "matches", matches,
                      "seconds", round(time.perf_counter()-started, 1), flush=True)
                batch = []
                if maximum and total_queries >= maximum:
                    break
        if batch:
            n_broad, n_final, n_match = process_batch(
                con, blocker, matcher, batch, candidate_writer, match_writer,
                pool, top_k, thresholds, default_threshold, model_feature_names)
            total_queries += len(batch)
            broad_pairs += n_broad
            final_pairs += n_final
            matches += n_match
            print("queries", total_queries, "broad_pairs", broad_pairs,
                  "final_candidates", final_pairs, "matches", matches,
                  "seconds", round(time.perf_counter()-started, 1), flush=True)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--broad-candidate", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--matcher-model", type=Path)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--country-threshold", action="append", default=[], metavar="COUNTRY:VALUE")
    args = parser.parse_args()
    thresholds = {}
    for value in args.country_threshold:
        country, threshold = value.rsplit(":", 1)
        thresholds[country] = float(threshold)
    run(args.data_dir, args.db, args.broad_candidate, args.candidate,
        args.matching, args.top_k, args.workers, args.batch_size,
        args.max_queries, thresholds, args.threshold, args.matcher_model,
        set(args.query_ids.read_text(encoding="utf-8").splitlines())
        if args.query_ids else None)
