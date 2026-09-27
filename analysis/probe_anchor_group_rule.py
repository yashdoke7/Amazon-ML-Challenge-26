"""Bounded post-match group-context rule test on fixed dev/frozen candidate pools.

Labels are used only for evaluation. The same preselected rule is applied to
frozen validation without fitting there. This script never changes outputs.
"""

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from anyascii import anyascii
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
DEV = TMP / "development_full"
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
RESULT = ROOT / "analysis/probe_anchor_group_rule_results.json"
GLOB = "*fresh_fast_charaddr_tfidf_top10_fresh_india_name_tfidf_top5_final.tsv"


def ids(value):
    return set(filter(None, (value or "").split(",")))


def read_lists(path, column):
    with path.open(encoding="utf-8", newline="") as stream:
        return {r["source1_entity_id"]: ids(r[column])
                for r in csv.DictReader(stream, delimiter="\t")}


def normalized(value):
    return " ".join(re.findall(r"[a-z0-9]+", anyascii(value or "").lower()))


def prepare():
    dev_final = sorted(DEV.glob(GLOB))
    assert len(dev_final) == 2, dev_final
    by_split = {}
    for split, path in (("development", dev_final[0]),
                        ("validation", dev_final[1])):
        assert ("validation" in path.name) == (split == "validation")
        base = TMP / ("fresh_dev_us_char_candidate_union.tsv"
                      if split == "development" else
                      "fresh_val_us_char_candidate_union.tsv")
        extra = TMP / ("fresh_india_name_char_fast_dev.tsv"
                       if split == "development" else
                       "fresh_india_name_char_fast_devval.tsv")
        candidates = read_lists(base, "candidate_entity_ids")
        with extra.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                q = row["source1_entity_id"]
                if q in candidates:
                    candidates[q].update(filter(None,
                        (row["candidate_entity_ids"] or "").split(",")[:5]))
        selected = read_lists(path, "matched_entity_ids")
        assert selected.keys() == candidates.keys()
        by_split[split] = (candidates, selected)
    wanted_q = set().union(*(c for c, _ in by_split.values()))
    truth = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in wanted_q:
                truth[q] = ids(row["matched_entity_ids"])
    assert len(truth) == len(wanted_q)
    wanted_t = set()
    for candidates, selected in by_split.values():
        for pool in candidates.values():
            wanted_t.update(pool)
        for group in selected.values():
            wanted_t.update(group)
    records = {}
    for source in (2, 3):
        with (TRAIN / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                t = row["entity_id"]
                if t in wanted_t:
                    name = normalized(row["business_name"])
                    addr = normalized(row["business_address"])
                    digits = {x for x in addr.split() if x.isdigit() and len(x) >= 2}
                    records[t] = (name, addr, digits)
    assert len(records) == len(wanted_t), (len(records), len(wanted_t))
    return by_split, truth, records


def features(candidates, selected, truth, records):
    out = {}
    selected_counts = {"groups_with_anchors": 0, "rejected_true_with_anchor": 0}
    for index, (q, pool) in enumerate(candidates.items(), 1):
        anchors = [records[t] for t in selected[q]]
        items = []
        if anchors:
            selected_counts["groups_with_anchors"] += 1
            for t in pool - selected[q]:
                name, addr, digits = records[t]
                if not name and not addr:
                    continue
                n = max((fuzz.token_set_ratio(name, a[0]) / 100
                         for a in anchors if name and a[0]), default=0.0)
                a = max((fuzz.token_set_ratio(addr, x[1]) / 100
                         for x in anchors if addr and x[1]), default=0.0)
                same_number = any(bool(digits & x[2]) for x in anchors)
                if t in truth[q]:
                    selected_counts["rejected_true_with_anchor"] += 1
                if n >= 0.75 and a >= 0.75:
                    items.append((n, a, same_number, t in truth[q]))
        out[q] = items
        if index % 5000 == 0:
            print("FEATURE_GROUPS", index, flush=True)
    return out, selected_counts


def f05(truth_count, pred_count, tp):
    if truth_count == 0:
        return float(pred_count == 0)
    if pred_count == 0:
        return 0.0
    return 1.25 * tp / (pred_count + 0.25 * truth_count)


def evaluate(candidates, selected, truth, scored, rule):
    n_min, a_min, require_number = rule
    score = base = 0.0
    added_true = added_false = changed = 0
    for q, items in scored.items():
        t = truth[q]
        p = selected[q]
        tp = len(t & p)
        before = f05(len(t), len(p), tp)
        extras = [hit for n, a, same_number, hit in items
                  if n >= n_min and a >= a_min and
                  (not require_number or same_number)]
        after = f05(len(t), len(p) + len(extras), tp + sum(extras))
        base += before
        score += after
        added_true += sum(extras)
        added_false += len(extras) - sum(extras)
        changed += int(after != before)
    return {"base": base / len(scored), "macro_f05": score / len(scored),
            "gain": (score - base) / len(scored),
            "added_true": added_true, "added_false": added_false,
            "changed_queries": changed}


def main():
    start = time.perf_counter()
    by_split, truth, records = prepare()
    print("TARGET_RECORDS", len(records), "SECONDS", round(time.perf_counter()-start, 1), flush=True)
    scored = {}
    counts = {}
    for split, (candidates, selected) in by_split.items():
        scored[split], counts[split] = features(candidates, selected, truth, records)
        print(split, counts[split], "SECONDS", round(time.perf_counter()-start, 1), flush=True)
    rules = [(n, a, number) for n in (0.8, 0.85, 0.9, 0.95, 0.98)
             for a in (0.8, 0.85, 0.9, 0.95, 0.98)
             for number in (False, True)]
    dev_c, dev_s = by_split["development"]
    dev_results = {str(rule): evaluate(dev_c, dev_s, truth, scored["development"], rule)
                   for rule in rules}
    best = max(rules, key=lambda r: dev_results[str(r)]["macro_f05"])
    val_c, val_s = by_split["validation"]
    report = {"groups": counts, "best_development_rule": list(best),
              "development": dev_results[str(best)],
              "frozen_validation": evaluate(val_c, val_s, truth, scored["validation"], best),
              "top_development_rules": sorted(dev_results.items(),
                  key=lambda x: x[1]["macro_f05"], reverse=True)[:8],
              "seconds": round(time.perf_counter()-start, 1)}
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
