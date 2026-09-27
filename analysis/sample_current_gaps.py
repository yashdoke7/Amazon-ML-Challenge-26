"""Inspect a seeded sample of current development misses; write supplied text locally."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def read_lists(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(filter(None, row[field].split(",")))
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    selected = read_lists(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    candidates = read_lists(DEV / "tfidf_top20_candidates.tsv", "candidate_entity_ids")
    truth = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in selected:
                truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    assert selected.keys() == candidates.keys() == truth.keys()

    rng = random.Random(20260927)
    samples = defaultdict(list)
    seen = defaultdict(int)
    for q in selected:
        for target in sorted(truth[q] - selected[q]):
            status = "rejected" if target in candidates[q] else "absent"
            key = (status, target[:2])
            seen[key] += 1
            bucket = samples[key]
            if len(bucket) < 15:
                bucket.append((q, target))
            else:
                j = rng.randrange(seen[key])
                if j < len(bucket):
                    bucket[j] = (q, target)
    picked = [pair for bucket in samples.values() for pair in bucket]
    qids = {q for q, _ in picked}
    tids = {t for _, t in picked}
    queries = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in qids:
                queries[row["entity_id"]] = row

    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.register("wanted", pd.DataFrame({"target_id": sorted(tids)}))
    targets = {row[0]: row[1:] for row in con.execute(
        "SELECT target_id,business_name,business_address,source FROM target WHERE target_id IN (SELECT target_id FROM wanted)"
    ).fetchall()}
    con.close()
    assert len(queries) == len(qids) and len(targets) == len(tids)
    output = ROOT / "analysis/current_gap_examples.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for q, t in picked:
            qrow = queries[q]
            tname, taddr, source = targets[t]
            status = "rejected" if t in candidates[q] else "absent"
            obj = {"status": status, "country": qrow["country"], "source": source,
                   "query_id": q, "target_id": t,
                   "query_name": qrow["business_name"], "target_name": tname,
                   "query_address": qrow["business_address"],
                   "target_address": taddr or "",
                   "selected_other_count": len(selected[q]),
                   "selected_other_ids": sorted(selected[q])[:3]}
            stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print("sampled", len(picked), "missed positives to", output,
          "population", dict(seen), flush=True)


if __name__ == "__main__":
    main()
