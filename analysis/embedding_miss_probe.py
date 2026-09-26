"""Compare embedding scores of reachable and unreachable validation true links."""

import csv
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from evaluate_first_matcher import sampled_truth

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
PAIRS = ROOT / "analysis" / "full_validation_combined_pairs.parquet"
DB = ROOT / ".duckdb" / "train_index.duckdb"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def cosine(frame, left, right, encoder):
    codes, texts = pd.factorize(pd.concat([frame[left], frame[right]], ignore_index=True), sort=False)
    vectors = encoder.encode(texts.tolist(), batch_size=128, device="cuda", show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True)
    n = len(frame)
    return np.einsum("ij,ij->i", vectors[codes[:n]], vectors[codes[n:]])


def summarize(frame, label):
    subset = frame[frame.status == label]
    return {"count": len(subset), "name_cosine_median": round(float(subset.name_cosine.median()), 4),
            "address_cosine_median": round(float(subset.address_cosine.median()), 4),
            "name_cosine_ge_08": round(float((subset.name_cosine >= 0.8).mean()), 4),
            "address_cosine_ge_08": round(float((subset.address_cosine >= 0.8).mean()), 4),
            "weak_name_count": int((subset.name_ratio < 50).sum()),
            "weak_name_address_cosine_ge_08": round(float((subset.loc[subset.name_ratio < 50, "address_cosine"] >= 0.8).mean()), 4)}


def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    torch.set_num_threads(4)
    started = time.perf_counter()
    truth = sampled_truth()
    candidates = pd.read_parquet(PAIRS)
    known = set(zip(candidates.s1_id, candidates.target_id))
    missing = [(s1, target) for s1, ids in truth.items() for target in ids if (s1, target) not in known]
    print("missing true pairs", len(missing), flush=True)
    query = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in truth:
                query[row["entity_id"]] = row
    con = duckdb.connect(str(DB), read_only=True)
    con.register("missing", pd.DataFrame(missing, columns=["s1_id", "target_id"]))
    targets = con.execute("""SELECT m.s1_id, m.target_id, t.business_name AS t_name,
        t.business_address AS t_address, t.country FROM missing m JOIN target t USING(target_id)""").df()
    con.close()
    assert len(targets) == len(missing)
    targets["q_name"] = [query[s]["business_name"] for s in targets.s1_id]
    targets["q_address"] = [query[s]["business_address"] for s in targets.s1_id]
    targets["status"] = "candidate_miss"
    found = candidates[candidates.is_match][["q_name", "t_name", "q_address", "t_address", "country"]].copy()
    found["status"] = "candidate_hit"
    frame = pd.concat([found, targets[["q_name", "t_name", "q_address", "t_address", "country", "status"]]],
                      ignore_index=True).fillna("")
    frame["name_ratio"] = [fuzz.ratio(a, b) for a, b in zip(frame.q_name, frame.t_name)]
    encoder = SentenceTransformer(MODEL, device="cuda")
    frame["name_cosine"] = cosine(frame, "q_name", "t_name", encoder)
    frame["address_cosine"] = cosine(frame, "q_address", "t_address", encoder)
    rank_candidates = candidates[["s1_id", "q_address", "t_address"]].copy().fillna("")
    rank_missing = targets[["s1_id", "q_address", "t_address"]].copy().fillna("")
    rank_frame = pd.concat([rank_candidates, rank_missing], ignore_index=True)
    rank_frame["address_cosine"] = cosine(rank_frame, "q_address", "t_address", encoder)
    existing = {s1: group.to_numpy() for s1, group in
                rank_frame.iloc[:len(rank_candidates)].groupby("s1_id").address_cosine}
    missing_ranks = [1 + int(np.sum(existing.get(s1, np.empty(0)) >= sim))
                     for s1, sim in rank_frame.iloc[len(rank_candidates):][["s1_id", "address_cosine"]]
                     .itertuples(index=False, name=None)]
    out = {"candidate_hit": summarize(frame, "candidate_hit"),
           "candidate_miss": summarize(frame, "candidate_miss"),
           "missing_by_country": frame[frame.status == "candidate_miss"].country.value_counts().to_dict(),
           "optimistic_address_rank_among_existing_candidates": {
               "median": int(np.median(missing_ranks)),
               "top_10": int(sum(rank <= 10 for rank in missing_ranks)),
               "top_50": int(sum(rank <= 50 for rank in missing_ranks)),
               "top_100": int(sum(rank <= 100 for rank in missing_ranks)),
               "top_200": int(sum(rank <= 200 for rank in missing_ranks))},
           "seconds": round(time.perf_counter()-started, 1),
           "limitation": "Missing positives were injected from ground truth; ranks exclude other targets outside the current candidate set and are optimistic, not global ANN recall."}
    (ROOT / "analysis" / "embedding_miss_probe_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
