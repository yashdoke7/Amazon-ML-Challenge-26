"""Stream the entire development pool through frozen blocker and matcher.

Checks whether a top-k preliminary learned block preserves reachable truths
and final matcher selections beyond the two 2k-query samples.
"""

import csv
import heapq
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from number_features import NUMBER_FEATURE_NAMES, number_pair_features  # noqa: E402
from validation import f05_for_query, entity_split  # noqa: E402
from infer import candidates  # noqa: E402

PACKAGE = ROOT / "code" / "business_entity_resolution"
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DB = ROOT / ".duckdb" / "train_index.duckdb"
CAPS = [20, 25, 30, 35, 40, 50]


def metric(truth, groups):
    tp = sum(len(ids & groups[q]) for q, ids in truth.items())
    fp = sum(len(groups[q] - ids) for q, ids in truth.items())
    return {"macro_f05": round(float(np.mean([
        f05_for_query(ids, groups[q]) for q, ids in truth.items()])), 6),
        "tp": tp, "fp": fp, "fn": sum(map(len, truth.values())) - tp}


def main():
    started = time.perf_counter()
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if entity_split(row["source1_entity_id"]) == "development":
                truth[row["source1_entity_id"]] = (
                    set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set())
    queries = []
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in truth:
                queries.append(row)
    assert len(queries) == len(truth) == 22_133
    blocker = joblib.load(PACKAGE / "model.joblib")
    matcher = joblib.load(PACKAGE / "number_model.joblib")
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=4")
    tops = defaultdict(list)
    full_matches = defaultdict(set)
    full_positive = 0
    total = 0
    for start in range(0, len(queries), 500):
        frame = candidates(con, queries[start:start+500], combined=True).fillna("")
        features = np.empty((len(frame), len(NUMBER_FEATURE_NAMES)), dtype=np.float32)
        for i, row in enumerate(frame.itertuples(index=False)):
            features[i] = number_pair_features(row.q_name, row.t_name,
                                                row.q_address, row.t_address,
                                                row.target_source)
        block_probs = blocker.predict_proba(features[:, :23])[:, 1]
        final_probs = matcher.predict_proba(features)[:, 1]
        s1 = frame.s1_id.to_numpy()
        targets = frame.target_id.to_numpy()
        labels = np.fromiter((target in truth[query] for query, target in zip(s1, targets)),
                             dtype=bool, count=len(frame))
        country_by_query = {row["entity_id"]: row["country"] for row in queries[start:start+500]}
        countries = np.fromiter((country_by_query[query] for query in s1),
                                dtype=object, count=len(frame))
        final_selected = final_probs >= np.where(countries == "India", 0.65, 0.75)
        for i in range(len(frame)):
            query, target = s1[i], targets[i]
            if final_selected[i]:
                full_matches[query].add(target)
            heap = tops[query]
            item = (float(block_probs[i]), target, bool(labels[i]), bool(final_selected[i]))
            if len(heap) < max(CAPS):
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
        full_positive += int(labels.sum())
        total += len(frame)
        if (start // 500) % 5 == 0:
            print("scored", start + len(queries[start:start+500]), "queries", total,
                  "pairs", round(time.perf_counter()-started, 1), "s", flush=True)
    con.close()
    result = {"queries": len(truth), "input_pairs": total,
              "input_positive_links": full_positive,
              "full_matcher": metric(truth, full_matches), "options": {}}
    for cap in CAPS:
        candidate_groups = defaultdict(set)
        selected_groups = defaultdict(set)
        count = 0
        for query, heap in tops.items():
            for _, target, is_true, is_selected in sorted(heap, reverse=True)[:cap]:
                count += 1
                if is_true:
                    candidate_groups[query].add(target)
                if is_selected:
                    selected_groups[query].add(target)
        result["options"][f"top{cap}"] = {
            "candidate_pairs": count,
            "positive_links": sum(map(len, candidate_groups.values())),
            "oracle": metric(truth, candidate_groups),
            "matcher": metric(truth, selected_groups),
            "estimated_test_zip_mb_at_5p7_bytes_per_pair": round(
                count / len(truth) * 1_732_544 * 5.7 / 1e6, 1),
        }
    result["seconds"] = round(time.perf_counter()-started, 1)
    destination = ROOT / "analysis" / "learned_blocking_full_dev_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
