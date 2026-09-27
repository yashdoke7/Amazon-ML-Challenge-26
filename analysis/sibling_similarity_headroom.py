"""Diagnostic: compare missed true targets to already selected true targets.

Positive-only upper bound; never use truth to make inference decisions.
"""

import csv
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import normalize  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def read(path, field, quota=None):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")[:quota])
                if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def norm(value):
    return normalize(anyascii(value or ""))


def similarity(left, right):
    return fuzz.ratio(left, right) if left and right else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("missing", "rejected"), default="missing")
    args = parser.parse_args()
    base = read(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    address = read(ROOT / "analysis/india_address_tfidf_dev_candidates.tsv",
                   "candidate_entity_ids", 20)
    selected = read(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    truth = read(DATA / "train_ground_truth.tsv", "matched_entity_ids")
    missed = []
    target_ids = set()
    for q, candidates in base.items():
        siblings = truth[q] & selected[q]
        for t in truth[q] - selected[q]:
            in_pool = t in (candidates | address.get(q, set()))
            if (in_pool == (args.stage == "rejected")) and siblings:
                missed.append((q, t, siblings))
                target_ids.add(t)
                target_ids.update(siblings)
    query_ids = {q for q, _, _ in missed}
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in query_ids:
                queries[row["entity_id"]] = (norm(row["business_name"]),
                                             norm(row["business_address"]))
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.register("wanted", pd.DataFrame({"target_id": list(target_ids)}))
    frame = con.execute("SELECT t.target_id,t.business_name,t.business_address "
                        "FROM wanted w JOIN target t USING(target_id)").df()
    con.close()
    targets = {r.target_id: (norm(r.business_name), norm(r.business_address))
               for r in frame.itertuples(index=False)}
    assert len(targets) == len(target_ids)
    counts = Counter()
    for q, t, siblings in missed:
        tn, ta = targets[t]
        qn, qa = queries[q]
        best_name = best_address = 0
        same_name = same_address = False
        for sibling in siblings:
            sn, sa = targets[sibling]
            best_name = max(best_name, similarity(sn, tn))
            best_address = max(best_address, similarity(sa, ta))
            same_name |= bool(sn and sn == tn)
            same_address |= bool(sa and sa == ta)
        q_name = similarity(qn, tn)
        q_address = similarity(qa, ta)
        counts["misses_with_selected_true_sibling"] += 1
        counts["sibling_exact_name"] += same_name
        counts["sibling_exact_address"] += same_address
        counts["sibling_either_exact"] += same_name or same_address
        for threshold in (60, 70, 80, 90, 95):
            counts[f"sibling_name_ge_{threshold}"] += best_name >= threshold
            counts[f"sibling_address_ge_{threshold}"] += best_address >= threshold
            counts[f"sibling_either_ge_{threshold}"] += max(best_name, best_address) >= threshold
        counts["sibling_name_better_than_query_10"] += best_name >= q_name + 10
        counts["sibling_address_better_than_query_10"] += best_address >= q_address + 10
    dest = ROOT / ("analysis/sibling_similarity_headroom_results.json" if
                   args.stage == "missing" else
                   "analysis/sibling_similarity_rejected_results.json")
    dest.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n",
                    encoding="utf-8")
    print(dest.read_text())


if __name__ == "__main__":
    main()
