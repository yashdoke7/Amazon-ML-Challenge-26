"""Group-level error budget for a saved development run.

This is a diagnostic comparator for a new design; it does not reuse model internals.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def parse_ids(value):
    return set(value.split(",")) if value else set()


def load_two_col(path, key, value, subset=None):
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if subset is None or row[key] in subset:
                result[row[key]] = parse_ids(row[value] or "")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path,
                        default=ROOT / "tmp/development_full/tfidf_top20_final.tsv")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "tmp/development_full/tfidf_top20_candidates.tsv")
    args = parser.parse_args()
    pred = load_two_col(args.matches, "source1_entity_id", "matched_entity_ids")
    cand = load_two_col(args.candidates, "source1_entity_id", "candidate_entity_ids")
    ids = set(pred)
    truth = load_two_col(TRAIN / "train_ground_truth.tsv", "source1_entity_id",
                         "matched_entity_ids", ids)
    country = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in ids:
                country[row["entity_id"]] = row["country"]
    groups = defaultdict(lambda: defaultdict(float))
    for s1 in ids:
        t, p, c = truth[s1], pred[s1], cand[s1]
        k = len(t)
        tp = len(p & t)
        fn_candidate = len(t - c)
        fn_model = len((t & c) - p)
        fp = len(p - t)
        score = 1.0 if not t and not p else (
            0.0 if not p else 1.25 * tp / (len(p) + 0.25 * k)
        )
        reachable = len(t & c)
        oracle = 1.0 if not t else (
            1.25 * reachable / (reachable + 0.25 * k) if reachable else 0.0
        )
        for label in ("all", f"country:{country[s1]}", f"group_size:{k}",
                      f"country:{country[s1]}:group_size:{k}"):
            g = groups[label]
            g["queries"] += 1
            g["score_sum"] += score
            g["oracle_sum"] += oracle
            g["exact_queries"] += int(p == t)
            g["candidate_complete_queries"] += int(t <= c)
            g["true_links"] += k
            g["tp"] += tp
            g["fp"] += fp
            g["fn_candidate"] += fn_candidate
            g["fn_model"] += fn_model
            g["queries_with_candidate_miss"] += int(fn_candidate > 0)
            g["queries_with_model_miss"] += int(fn_model > 0)
            g["queries_with_fp"] += int(fp > 0)
    report = {}
    for key, d in groups.items():
        report[key] = {k: int(v) for k, v in d.items()
                       if k not in ("score_sum", "oracle_sum")}
        report[key]["macro_f0_5"] = d["score_sum"] / d["queries"]
        report[key]["candidate_oracle"] = d["oracle_sum"] / d["queries"]
    path = ROOT / "analysis/fresh_error_budget_results.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if ":group_size:" not in k},
                     indent=2))


if __name__ == "__main__":
    main()
