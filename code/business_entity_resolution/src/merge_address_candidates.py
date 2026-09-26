"""Score bounded address-retrieved India pairs and merge with top-40 results.

The output candidate TSV is the exact union passed to the final matcher.
Rows from the base two-stage scorer and the India-only address TSV must follow
the Source 1 input order; this is checked rather than inferred from IDs.
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

from generalized_features import GENERALIZED_FEATURE_NAMES
from model_features import extractor_for
from rescore_candidates import featurize_rows


def process_batch(con, matcher, specialist, pool, batch,
                  candidate_writer, match_writer, blank_threshold):
    queries = {q["entity_id"]: q for q, _, _, _ in batch}
    pairs = []
    for q, base, _, extra in batch:
        existing = set(base)
        if len(existing) != len(base) or len(set(extra)) != len(extra):
            raise ValueError("Duplicate candidate ID")
        pairs.extend((q["entity_id"], target) for target in extra
                     if target not in existing)
    selected = {q["entity_id"]: set() for q, _, _, _ in batch}
    if pairs:
        con.register("address_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id"]))
        frame = con.execute("""SELECT p.s1_id,p.target_id,t.business_name t_name,
            coalesce(t.business_address,'') t_address,t.source target_source,
            t.country target_country,f.df target_core_df
            FROM address_pairs p JOIN target t USING(target_id)
            LEFT JOIN blank_target_frequency f USING(target_id)""").df()
        con.unregister("address_pairs")
        if len(frame) != len(pairs) or not frame.target_country.eq("India").all():
            raise ValueError("Address target missing or outside India")
        feature_rows = [(queries[row.s1_id]["business_name"], row.t_name,
                         queries[row.s1_id]["business_address"], row.t_address,
                         row.target_source) for row in frame.itertuples(index=False)]
        chunks = (feature_rows[i:i+50_000] for i in range(0, len(feature_rows), 50_000))
        features = np.concatenate(list(pool.map(
            partial(featurize_rows, model_feature_names=GENERALIZED_FEATURE_NAMES),
            chunks)))
        probabilities = matcher.predict_proba(features)[:, 1]
        blank = frame.t_address.eq("").to_numpy()
        if blank.any():
            frequencies = frame.loc[blank, "target_core_df"]
            if frequencies.isna().any():
                raise ValueError("Blank-address target lacks name frequency")
            special_features = np.empty((int(blank.sum()), 36), dtype=np.float32)
            special_features[:, :35] = features[blank]
            special_features[:, 35] = np.log1p(frequencies.to_numpy(dtype=np.float32))
            probabilities[blank] = specialist.predict_proba(special_features)[:, 1]
        thresholds = np.where(blank, blank_threshold, 0.65)
        for (s1, target), keep in zip(frame[["s1_id", "target_id"]].itertuples(
                index=False, name=None), probabilities >= thresholds):
            if keep:
                selected[s1].add(target)
    for q, base, raw, extra in batch:
        s1 = q["entity_id"]
        candidate_writer.writerow([s1, ",".join(sorted(set(base) | set(extra)))])
        match_writer.writerow([s1, ",".join(sorted(set(raw) | selected[s1]))])
    return len(pairs), sum(map(len, selected.values()))


def run(data_dir, db_path, base_candidate, base_matching, extra_path,
        candidate_path, matching_path, matcher_path, specialist_path,
        query_ids=None, workers=4, batch_size=5000, blank_threshold=0.8):
    paths = [base_candidate, base_matching, extra_path, candidate_path, matching_path]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Input and output paths must be distinct")
    matcher = joblib.load(matcher_path)
    specialist = joblib.load(specialist_path)
    names, _ = extractor_for(matcher)
    if names != GENERALIZED_FEATURE_NAMES or list(specialist.feature_name_) != (
            GENERALIZED_FEATURE_NAMES + ["log_target_core_frequency"]):
        raise ValueError("Unexpected model feature contract")
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    started = time.perf_counter()
    total_queries = total_pairs = total_selected = 0
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    source1 = data_dir / f"{data_dir.name}_source1.tsv"
    with ProcessPoolExecutor(max_workers=workers) as pool, \
         source1.open(encoding="utf-8", newline="") as q_stream, \
         base_candidate.open(encoding="utf-8", newline="") as bc_stream, \
         base_matching.open(encoding="utf-8", newline="") as bm_stream, \
         extra_path.open(encoding="utf-8", newline="") as extra_stream, \
         candidate_path.open("w", encoding="utf-8", newline="") as c_stream, \
         matching_path.open("w", encoding="utf-8", newline="") as m_stream:
        queries = csv.DictReader(q_stream, delimiter="\t")
        if query_ids is not None:
            queries = (q for q in queries if q["entity_id"] in query_ids)
        base_candidates = csv.DictReader(bc_stream, delimiter="\t")
        base_matches = csv.DictReader(bm_stream, delimiter="\t")
        extras = csv.DictReader(extra_stream, delimiter="\t")
        if (base_candidates.fieldnames != ["source1_entity_id", "candidate_entity_ids"]
                or base_matches.fieldnames != ["source1_entity_id", "matched_entity_ids"]
                or extras.fieldnames != ["source1_entity_id", "candidate_entity_ids"]):
            raise ValueError("Unexpected input header")
        next_extra = next(extras, None)
        candidate_writer = csv.writer(c_stream, delimiter="\t", lineterminator="\n")
        match_writer = csv.writer(m_stream, delimiter="\t", lineterminator="\n")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        batch = []
        for q, base, raw in itertools.zip_longest(queries, base_candidates, base_matches):
            if q is None or base is None or raw is None:
                raise ValueError("Base candidate/matching row counts differ")
            s1 = q["entity_id"]
            if s1 != base["source1_entity_id"] or s1 != raw["source1_entity_id"]:
                raise ValueError("Base rows differ from Source 1 order")
            extra = []
            if q["country"] == "India":
                if next_extra is None or next_extra["source1_entity_id"] != s1:
                    raise ValueError("India address row differs from Source 1 order")
                extra = next_extra["candidate_entity_ids"].split(",") if next_extra[
                    "candidate_entity_ids"] else []
                next_extra = next(extras, None)
            base_ids = base["candidate_entity_ids"].split(",") if base[
                "candidate_entity_ids"] else []
            raw_ids = raw["matched_entity_ids"].split(",") if raw[
                "matched_entity_ids"] else []
            if not set(raw_ids) <= set(base_ids):
                raise ValueError("Base selected ID outside candidate set")
            batch.append((q, base_ids, raw_ids, extra))
            if len(batch) >= batch_size:
                n_pairs, n_selected = process_batch(
                    con, matcher, specialist, pool, batch,
                    candidate_writer, match_writer, blank_threshold)
                total_queries += len(batch)
                total_pairs += n_pairs
                total_selected += n_selected
                batch.clear()
                if total_queries % 50_000 == 0:
                    print("queries", total_queries, "new_pairs", total_pairs,
                          "selected", total_selected,
                          "seconds", round(time.perf_counter()-started, 1), flush=True)
        if batch:
            n_pairs, n_selected = process_batch(
                con, matcher, specialist, pool, batch,
                candidate_writer, match_writer, blank_threshold)
            total_queries += len(batch)
            total_pairs += n_pairs
            total_selected += n_selected
        if next_extra is not None:
            raise ValueError("Unused India address rows")
    con.close()
    print("COMPLETE", total_queries, total_pairs, total_selected,
          "seconds", round(time.perf_counter()-started, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--base-matching", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--blank-specialist", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--blank-threshold", type=float, default=0.8)
    args = parser.parse_args()
    run(args.data_dir, args.db, args.base_candidate, args.base_matching,
        args.extra, args.candidate, args.matching, args.model,
        args.blank_specialist,
        set(args.query_ids.read_text(encoding="utf-8").splitlines())
        if args.query_ids else None,
        args.workers, args.batch_size, args.blank_threshold)
