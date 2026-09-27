"""Raw-field signal coverage among development true links absent from candidates."""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from anyascii import anyascii
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"


def load_lists(path, field, subset=None):
    out = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if subset is None or row["source1_entity_id"] in subset:
                out[row["source1_entity_id"]] = set(filter(None, (row[field] or "").split(",")))
    return out


def tokens(value):
    return re.findall(r"[a-z0-9]+", anyascii(value or "").lower())


def main():
    candidates = load_lists(DEV / "tfidf_top20_candidates.tsv", "candidate_entity_ids")
    selected = load_lists(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    truth = load_lists(TRAIN / "train_ground_truth.tsv", "matched_entity_ids", set(candidates))
    missing = defaultdict(list)
    for q, t in truth.items():
        for target in t - candidates[q]:
            missing[target].append((q, "absent_true"))
        for target in (t & candidates[q]) - selected[q]:
            missing[target].append((q, "rejected_true"))
        for target in selected[q] - t:
            missing[target].append((q, "selected_false"))
    qids = {q for group in missing.values() for q, _ in group}
    qrows = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in qids:
                qrows[row["entity_id"]] = row
    stats = defaultdict(Counter)
    for source in (2, 3):
        with (TRAIN / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["entity_id"] not in missing:
                    continue
                for q, status in missing[row["entity_id"]]:
                    s1 = qrows[q]
                    name1, name2 = tokens(s1["business_name"]), tokens(row["business_name"])
                    addr1, addr2 = tokens(s1["business_address"]), tokens(row["business_address"])
                    nums1 = {x for x in addr1 if x.isdigit() and len(x) >= 2}
                    nums2 = {x for x in addr2 if x.isdigit() and len(x) >= 2}
                    name_sim = fuzz.token_set_ratio(" ".join(name1), " ".join(name2)) / 100
                    addr_sim = (fuzz.token_set_ratio(" ".join(addr1), " ".join(addr2)) / 100
                                if addr2 else 0.0)
                    s = stats[f"{status}:{s1['country']}:{source}"]
                    s["n"] += 1
                    s["target_address_empty"] += int(not addr2)
                    s["shared_2plus_digit"] += int(bool(nums1 & nums2))
                    s["shared_address_token"] += int(bool(set(addr1) & set(addr2)))
                    s["name_sim_ge_0p8"] += int(name_sim >= 0.8)
                    s["addr_sim_ge_0p8"] += int(addr_sim >= 0.8)
                    s["either_sim_ge_0p8"] += int(name_sim >= 0.8 or addr_sim >= 0.8)
                    s["both_sim_lt_0p6"] += int(name_sim < 0.6 and addr_sim < 0.6)
    output = ROOT / "analysis/fresh_missing_signals_results.json"
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
