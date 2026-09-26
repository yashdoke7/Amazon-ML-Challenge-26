"""Bound the value of using a correct predicted target as a second query.

This is a labeled development feasibility probe, not a deployable result:
it selects one *known-correct* anchor among current predictions, so recovery
is an optimistic upper bound. It never trains on development labels.
"""

import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from infer import candidates  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
TMP = ROOT / "tmp" / "development_full"
LIMIT = 1000
BATCH = 100


def read_ids(path, column):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(row[column].split(","))
                if row[column] else set()
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    started = time.perf_counter()
    selected = read_ids(TMP / "generalized_exclusive.tsv", "matched_entity_ids")
    broad = read_ids(TMP / "candidate_pairs.tsv", "candidate_entity_ids")
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in selected:
                truth[q] = set(row["matched_entity_ids"].split(",")) \
                    if row["matched_entity_ids"] else set()
    eligible = [q for q in selected if (selected[q] & truth[q]) and (truth[q] - broad[q])]
    sample = random.Random(20260926).sample(eligible, min(LIMIT, len(eligible)))
    anchors = {q: sorted(selected[q] & truth[q])[0] for q in sample}
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"), read_only=True)
    con.execute("SET threads=4")
    con.register("anchor_ids", pd.DataFrame({"target_id": list(anchors.values())}))
    details = con.execute("""SELECT t.target_id entity_id,t.business_name,t.business_address,t.country
        FROM anchor_ids a JOIN target t USING(target_id)""").df()
    con.unregister("anchor_ids")
    assert len(details) == len(anchors)
    owner = {target: q for q, target in anchors.items()}
    counts = Counter()
    counts["eligible_groups"] = len(eligible)
    counts["sampled_groups"] = len(sample)
    counts["missed_true_links"] = sum(len(truth[q] - broad[q]) for q in sample)
    for first in range(0, len(details), BATCH):
        rows = details.iloc[first:first+BATCH].to_dict("records")
        frame = candidates(con, rows, combined=True)
        counts["second_hop_pairs"] += len(frame)
        recovered = {q: set() for q in (owner[row["entity_id"]] for row in rows)}
        for anchor, target in frame[["s1_id", "target_id"]].itertuples(index=False, name=None):
            q = owner[anchor]
            if target not in broad[q]:
                recovered[q].add(target)
        counts["new_pairs"] += sum(map(len, recovered.values()))
        counts["new_true_links"] += sum(len(recovered[q] & (truth[q] - broad[q])) for q in recovered)
        counts["groups_with_recovery"] += sum(bool(recovered[q] & (truth[q] - broad[q])) for q in recovered)
        print("anchors", min(first+BATCH, len(details)), "new_true", counts["new_true_links"],
              "new_pairs", counts["new_pairs"], "seconds", round(time.perf_counter()-started, 1), flush=True)
    con.close()
    result = dict(counts)
    result["optimistic_missing_link_recall"] = round(
        counts["new_true_links"] / max(1, counts["missed_true_links"]), 6)
    result["seconds"] = round(time.perf_counter()-started, 1)
    destination = ROOT / "analysis" / "anchor_retrieval_probe_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
