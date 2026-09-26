"""Inspect remaining errors of the promoted matcher on development records.

Aggregate JSON is safe for handoff; individual supplied records are printed
only to the local ignored log for manual inspection.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from number_features import first_number  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"


def read_ids(path, column):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[column].split(","))
                if row[column] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    selected = read_ids(DEV / "hard_negative_top40_final.tsv", "matched_entity_ids")
    candidates = read_ids(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in selected:
                truth[q] = set(row["matched_entity_ids"].split(",")) \
                    if row["matched_entity_ids"] else set()
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in selected:
                queries[row["entity_id"]] = row
    assert set(selected) == set(candidates) == set(truth) == set(queries)
    pairs = []
    for q in truth:
        for target in truth[q] | selected[q]:
            status = ("tp" if target in truth[q] and target in selected[q] else
                      "fp" if target in selected[q] else
                      "fn_rejected" if target in candidates[q] else "fn_absent")
            pairs.append((q, target, status))
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"), read_only=True)
    con.register("audit_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id", "status"]))
    frame = con.execute("""SELECT p.s1_id,p.target_id,p.status,t.business_name,t.business_address,t.source
        FROM audit_pairs p JOIN target t USING(target_id)""").df().fillna("")
    con.close()
    assert len(frame) == len(pairs)
    features = np.empty((len(frame), len(GENERALIZED_FEATURE_NAMES)), dtype=np.float32)
    stats = defaultdict(Counter)
    for i, row in enumerate(frame.itertuples(index=False)):
        q = queries[row.s1_id]
        values = generalized_pair_features(q["business_name"], row.business_name,
                                           q["business_address"], row.business_address,
                                           row.source)
        features[i] = values
        key = row.status
        info = stats[key]
        info["total"] += 1
        info[f"country_{q['country']}"] += 1
        info[f"source_{row.source}"] += 1
        info["target_blank_address"] += not bool(row.business_address)
        info["name_ratio_lt_40"] += values[0] < 40
        info["name_ratio_lt_60"] += values[0] < 60
        info["address_ratio_lt_40"] += values[8] < 40
        info["address_ratio_lt_60"] += values[8] < 60
        info["both_ratio_lt_60"] += values[0] < 60 and values[8] < 60
        info["ascii_name_ratio_ge_75"] += values[27] >= 75
        info["name_nonascii"] += bool(values[33])
        qnum, tnum = first_number(q["business_address"]), first_number(row.business_address)
        info["numbers_present"] += qnum >= 0 and tnum >= 0
        info["number_mismatch"] += qnum >= 0 and tnum >= 0 and qnum != tnum
    model = joblib.load(ROOT / "code" / "business_entity_resolution" / "generalized_model.joblib")
    probabilities = model.predict_proba(features)[:, 1]
    summaries = {}
    for status, counts in stats.items():
        mask = frame.status.to_numpy() == status
        summaries[status] = {**dict(counts),
            "probability_p10_p50_p90": [round(float(x), 4) for x in
                np.percentile(probabilities[mask], [10, 50, 90])]}
    destination = ROOT / "analysis" / "hard_negative_error_audit_results.json"
    destination.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    for status in ("fn_absent", "fn_rejected", "fp"):
        indices = np.flatnonzero(frame.status.to_numpy() == status)
        indices = indices[np.argsort(-probabilities[indices], kind="stable")[:12]]
        print("\nSAMPLES", status, flush=True)
        for i in indices:
            row = frame.iloc[i]
            query = queries[row.s1_id]
            print(json.dumps({"query": row.s1_id, "target": row.target_id,
                              "country": query["country"], "source": row.source,
                              "probability": round(float(probabilities[i]), 4),
                              "q_name": query["business_name"], "t_name": row.business_name,
                              "q_address": query["business_address"],
                              "t_address": row.business_address}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
