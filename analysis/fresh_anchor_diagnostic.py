"""Test if other predicted records reveal matches weak against Source 1.

Uses only raw record text plus labels for a bounded development diagnostic.
"""

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import anyascii
import numpy as np
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def load_lists(path, field, subset=None):
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if subset is None or row["source1_entity_id"] in subset:
                result[row["source1_entity_id"]] = set(filter(None, (row[field] or "").split(",")))
    return result


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", anyascii.anyascii(s or "").lower()))


def sim(a, b):
    return fuzz.token_set_ratio(a, b) / 100 if a and b else 0.0


def summarize(items):
    result = {"n": len(items)}
    for field in ("query_name", "anchor_name", "query_address", "anchor_address"):
        x = np.array([v[field] for v in items])
        result[field] = {"median": float(np.median(x)), "p90": float(np.quantile(x, 0.9))}
    result["anchor_stronger_name_0p15"] = sum(
        v["anchor_name"] > v["query_name"] + 0.15 for v in items
    )
    result["anchor_stronger_address_0p15"] = sum(
        v["anchor_address"] > v["query_address"] + 0.15 for v in items
    )
    result["anchor_name_high_query_low"] = sum(
        v["anchor_name"] >= 0.9 and v["query_name"] < 0.75 for v in items
    )
    result["anchor_address_high_query_low"] = sum(
        v["anchor_address"] >= 0.9 and v["query_address"] < 0.75 for v in items
    )
    return result


def main():
    rng = random.Random(260927)
    selected = load_lists(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    candidates = load_lists(DEV / "tfidf_top20_candidates.tsv", "candidate_entity_ids")
    truth = load_lists(TRAIN / "train_ground_truth.tsv", "matched_entity_ids", set(selected))
    buckets = defaultdict(list)
    seen = defaultdict(int)
    for q in selected:
        if not selected[q]:
            continue
        for t in truth[q] - selected[q]:
            status = "rejected_positive" if t in candidates[q] else "absent_positive"
            key = (status, t[:2])
            seen[key] += 1
            bucket = buckets[key]
            if len(bucket) < 500:
                bucket.append((q, t))
            else:
                i = rng.randrange(seen[key])
                if i < len(bucket):
                    bucket[i] = (q, t)
        false_candidates = list(candidates[q] - truth[q] - selected[q])
        if false_candidates:
            t = rng.choice(false_candidates)
            key = ("rejected_negative", t[:2])
            seen[key] += 1
            bucket = buckets[key]
            if len(bucket) < 500:
                bucket.append((q, t))
            else:
                i = rng.randrange(seen[key])
                if i < len(bucket):
                    bucket[i] = (q, t)
    samples = [(kind, source, q, t) for (kind, source), bucket in buckets.items()
               for q, t in bucket]
    qids = {q for _, _, q, _ in samples}
    tids = {t for _, _, _, t in samples}
    for q in qids:
        tids.update(selected[q])
    query = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in qids:
                query[row["entity_id"]] = (norm(row["business_name"]), norm(row["business_address"]))
    target = {}
    for source in (2, 3):
        with (TRAIN / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["entity_id"] in tids:
                    target[row["entity_id"]] = (norm(row["business_name"]),
                                                norm(row["business_address"]))
    scores = defaultdict(list)
    for kind, source, q, t in samples:
        qname, qaddr = query[q]
        tname, taddr = target[t]
        anchors = [target[a] for a in selected[q] if a in target]
        scores[f"{kind}:{source}"].append({
            "query_name": sim(qname, tname),
            "anchor_name": max((sim(n, tname) for n, _ in anchors), default=0),
            "query_address": sim(qaddr, taddr),
            "anchor_address": max((sim(a, taddr) for _, a in anchors), default=0),
        })
    report = {key: summarize(value) for key, value in scores.items()}
    report["sample_counts"] = {f"{a}:{b}": n for (a, b), n in seen.items()}
    output = ROOT / "analysis/fresh_anchor_diagnostic_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
