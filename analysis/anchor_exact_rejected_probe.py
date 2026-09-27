"""Evaluate exact target-to-target evidence among already retrieved but rejected pairs.

Anchors are current predictions only. Labels are used only after candidate
selection to measure macro F0.5. No external data or test labels are used.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from validation import f05_for_query  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def read(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[field].split(",")) if row[field] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"),
                        default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    stem = ("tfidf_top20" if args.split == "development" else
            "validation_tfidf_merged")
    pool = read(DEV / f"{stem}_candidates.tsv", "candidate_entity_ids")
    chosen = read(DEV / f"{stem}_final.tsv", "matched_entity_ids")
    if set(pool) != set(chosen):
        raise ValueError("Candidate and selected query sets differ")
    truth = read(DATA / "train_ground_truth.tsv", "matched_entity_ids")
    candidate_rows = [(q, t) for q, ids in pool.items()
                      for t in ids - chosen[q]]
    anchor_rows = [(q, t) for q, ids in chosen.items() for t in ids]
    print("queries", len(pool), "rejected_candidate_pairs", len(candidate_rows),
          "predicted_anchors", len(anchor_rows), flush=True)
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.execute("SET memory_limit='16GB'")
    con.execute("SET threads=8")
    con.register("candidate_pairs", pd.DataFrame(candidate_rows,
                 columns=["q", "candidate_id"]))
    con.register("anchor_pairs", pd.DataFrame(anchor_rows,
                 columns=["q", "anchor_id"]))
    frame = con.execute("""
        WITH c AS (
            SELECT p.q,p.candidate_id,k.core_key,
                   lower(trim(coalesce(t.business_address,''))) addr
            FROM candidate_pairs p JOIN target_keys k ON p.candidate_id=k.target_id
            JOIN target t ON p.candidate_id=t.target_id
        ), a AS (
            SELECT p.q,p.anchor_id,k.core_key,
                   lower(trim(coalesce(t.business_address,''))) addr
            FROM anchor_pairs p JOIN target_keys k ON p.anchor_id=k.target_id
            JOIN target t ON p.anchor_id=t.target_id
        )
        SELECT c.q,c.candidate_id,
               bool_or(c.core_key!='' AND c.core_key=a.core_key) core_exact,
               bool_or(c.addr!='' AND c.addr=a.addr) address_exact
        FROM c JOIN a USING(q)
        WHERE (c.core_key!='' AND c.core_key=a.core_key)
           OR (c.addr!='' AND c.addr=a.addr)
        GROUP BY c.q,c.candidate_id
    """).df()
    con.close()
    baseline = sum(f05_for_query(truth[q], chosen[q]) for q in pool) / len(pool)
    results = {}
    for label, mask in (
            ("core_exact", frame.core_exact),
            ("address_exact", frame.address_exact),
            ("either_exact", frame.core_exact | frame.address_exact),
            ("both_exact", frame.core_exact & frame.address_exact)):
        eligible = frame.loc[mask]
        additions = defaultdict(set)
        for q, target in eligible[["q", "candidate_id"]].itertuples(index=False, name=None):
            additions[q].add(target)
        score = sum(f05_for_query(truth[q], chosen[q] | additions[q]) for q in pool) / len(pool)
        tp = sum(len(ids & truth[q]) for q, ids in additions.items())
        results[label] = {"new_pairs": len(eligible), "true_pairs": tp,
                          "false_pairs": len(eligible) - tp,
                          "macro_f05": score, "gain": score - baseline}
    result = {"split": args.split, "queries": len(pool),
              "candidate_pairs_examined": len(candidate_rows),
              "anchors": len(anchor_rows), "baseline_macro_f05": baseline,
              "results": results, "seconds": round(time.perf_counter()-started, 1),
              "note": "Before cap and exclusive ownership; validation is for one frozen development rule only."}
    dest = args.output or ROOT / f"analysis/anchor_exact_{args.split}_results.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
