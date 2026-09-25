"""Reproducible row-level exploration of supplied 2026 entity-resolution TSVs.

Samples labeled entities uniformly, then reads the source files once. This script
does not use external identities or train a model.
"""
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import regex
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
OUT = Path(__file__).resolve().parent / "deep_eda_results.json"
RNG = random.Random(20260925)
N_SAMPLE = 30000
TOKEN = regex.compile(r"[\p{L}\p{M}\p{N}]+")
NUM = regex.compile(r"\p{N}+")
DOMAIN = re.compile(r"(?:\bwww\.|\b[a-z0-9-]+\.(?:com|org|net|in|fr|co|io)\b)", re.I)
PLACEHOLDER = re.compile(r"\b(?:null|none|n/?a|unknown)\b", re.I)
SCRIPTS = {s: regex.compile(rf"\p{{Script={s}}}") for s in ("Latin", "Devanagari", "Tamil", "Kannada", "Telugu", "Bengali", "Gujarati", "Gurmukhi", "Malayalam", "Arabic")}


def basic(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(TOKEN.findall(s))


def scripts(s):
    found = [name for name, rx in SCRIPTS.items() if rx.search(s)]
    if not found:
        return "Other"
    return "+".join(found)


def reservoir_truth():
    sample = []
    path = ROOT / "train" / "train_ground_truth.tsv"
    with path.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            item = (row["source1_entity_id"], row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else [])
            if i < N_SAMPLE:
                sample.append(item)
            else:
                j = RNG.randrange(i + 1)
                if j < N_SAMPLE:
                    sample[j] = item
    return sample


def scan_train(sample):
    wanted = {s1 for s1, _ in sample}
    for _, ids in sample:
        wanted.update(ids)
    records = {}
    full = {}
    for source in (1, 2, 3):
        stats = Counter()
        long_name = []
        long_address = []
        with (ROOT / "train" / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                stats["rows"] += 1
                name, address = row["business_name"], row["business_address"]
                stats[f"country:{row['country']}"] += 1
                stats["missing_name"] += not name.strip()
                stats["missing_address"] += not address.strip()
                stats["placeholder_name"] += bool(PLACEHOLDER.search(name))
                stats["placeholder_address"] += bool(PLACEHOLDER.search(address))
                stats["domain_name"] += bool(DOMAIN.search(name))
                stats["non_latin_name"] += bool(SCRIPTS["Devanagari"].search(name) or SCRIPTS["Tamil"].search(name) or SCRIPTS["Kannada"].search(name) or SCRIPTS["Telugu"].search(name) or SCRIPTS["Bengali"].search(name) or SCRIPTS["Gujarati"].search(name) or SCRIPTS["Gurmukhi"].search(name) or SCRIPTS["Malayalam"].search(name))
                stats["leading_symbol_name"] += bool(name and not name[0].isalnum())
                stats["name_over_100_chars"] += len(name) > 100
                stats["address_over_200_chars"] += len(address) > 200
                if len(name) > 150 and len(long_name) < 3:
                    long_name.append(row)
                if len(address) > 300 and len(long_address) < 3:
                    long_address.append(row)
                if row["entity_id"] in wanted:
                    records[row["entity_id"]] = row
        full[f"train_source{source}"] = {"counts": stats, "long_name_examples": long_name, "long_address_examples": long_address}
        print(f"scanned train source {source}", flush=True)
    assert len(records) == len(wanted), (len(records), len(wanted))
    return records, full


def scan_france():
    out = {}
    for source in (1, 2, 3):
        counts = Counter()
        examples = []
        with (ROOT / "test" / f"test_source{source}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["country"] != "France":
                    continue
                counts["rows"] += 1
                name, address = row["business_name"], row["business_address"]
                counts["missing_address"] += not address.strip()
                counts["name_with_accent"] += any(unicodedata.category(ch) == "Mn" or (ord(ch) > 127 and "LATIN" in unicodedata.name(ch, "")) for ch in name)
                counts["address_with_accent"] += any(ord(ch) > 127 and "LATIN" in unicodedata.name(ch, "") for ch in address)
                counts["name_domain"] += bool(DOMAIN.search(name))
                counts["name_sarl_sas_sa"] += bool(re.search(r"\b(?:sarl|sas|s\.a\.)\b", name, re.I))
                counts["address_5_digit_number"] += bool(re.search(r"\b\d{5}\b", address))
                if len(examples) < 10:
                    examples.append(row)
                else:
                    j = RNG.randrange(counts["rows"])
                    if j < 10:
                        examples[j] = row
        out[f"test_source{source}_France"] = {"counts": counts, "examples": examples}
        print(f"scanned test France source {source}", flush=True)
    return out


def pair_profile(sample, records):
    overall = Counter()
    by = defaultdict(Counter)
    buckets = Counter()
    cases = defaultdict(list)
    for s1_id, ids in sample:
        s1 = records[s1_id]
        for target_id in ids:
            t = records[target_id]
            n1, n2 = s1["business_name"], t["business_name"]
            a1, a2 = s1["business_address"], t["business_address"]
            nn1, nn2 = basic(n1), basic(n2)
            aa1, aa2 = basic(a1), basic(a2)
            name_sim = fuzz.token_sort_ratio(nn1, nn2)
            addr_sim = fuzz.token_sort_ratio(aa1, aa2) if aa2 else 0
            name_set = fuzz.token_set_ratio(nn1, nn2)
            addr_set = fuzz.token_set_ratio(aa1, aa2) if aa2 else 0
            nums1, nums2 = set(NUM.findall(a1)), set(NUM.findall(a2))
            source = target_id[:2]
            country = s1["country"]
            flags = {
                "pairs": 1,
                "raw_exact_name": n1.casefold() == n2.casefold(),
                "basic_exact_name": bool(nn1) and nn1 == nn2,
                "raw_exact_address": bool(a2) and a1.casefold() == a2.casefold(),
                "basic_exact_address": bool(aa2) and aa1 == aa2,
                "missing_target_address": not a2.strip(),
                "name_below_30": name_sim < 30,
                "name_below_50": name_sim < 50,
                "name_below_70": name_sim < 70,
                "address_below_30": addr_sim < 30,
                "address_below_50": addr_sim < 50,
                "address_below_70": addr_sim < 70,
                "both_below_50": name_sim < 50 and addr_sim < 50,
                "token_set_name_90_plus": name_set >= 90,
                "token_set_address_90_plus": addr_set >= 90,
                "different_name_scripts": scripts(n1) != scripts(n2),
                "target_domain_name": bool(DOMAIN.search(n2)),
                "both_have_address_numbers": bool(nums1 and nums2),
                "address_numbers_overlap": bool(nums1 & nums2),
                "disjoint_address_numbers": bool(nums1 and nums2 and not nums1 & nums2),
                "country_mismatch": country != t["country"],
            }
            for key, value in flags.items():
                overall[key] += value
                by[f"{country}/{source}"][key] += value
            name_bin = "<30" if name_sim < 30 else "30-49" if name_sim < 50 else "50-69" if name_sim < 70 else "70-89" if name_sim < 90 else "90+"
            addr_bin = "missing" if not aa2 else "<30" if addr_sim < 30 else "30-49" if addr_sim < 50 else "50-69" if addr_sim < 70 else "70-89" if addr_sim < 90 else "90+"
            buckets[f"name:{name_bin}/address:{addr_bin}"] += 1
            cats = []
            if name_sim < 30 and addr_sim >= 70:
                cats.append("name_unrelated_address_strong")
            if addr_sim < 30 and name_sim >= 70:
                cats.append("address_unrelated_name_strong")
            if name_sim < 50 and addr_sim < 50:
                cats.append("both_weak")
            if scripts(n1) != scripts(n2):
                cats.append("script_change")
            if nums1 and nums2 and not nums1 & nums2 and addr_sim >= 70:
                cats.append("number_disagreement_despite_similar_address")
            if DOMAIN.search(n2):
                cats.append("domain_alias")
            if not a2.strip():
                cats.append("missing_address")
            if nn1 == nn2 and aa1 != aa2:
                cats.append("same_normalized_name_changed_address")
            if aa1 == aa2 and nn1 != nn2:
                cats.append("same_normalized_address_changed_name")
            for cat in cats:
                if len(cases[cat]) < 5:
                    cases[cat].append({"s1": s1, "target": t, "name_sim": round(name_sim, 1), "address_sim": round(addr_sim, 1)})
    return {"overall": overall, "by_country_source": by, "similarity_grid": buckets, "cases": cases}


def main():
    sample = reservoir_truth()
    print(f"selected {len(sample)} training entities", flush=True)
    records, full = scan_train(sample)
    profile = pair_profile(sample, records)
    france = scan_france()
    result = {"sample_s1_count": len(sample), "sample_truth_edges": profile["overall"]["pairs"], "train_record_profile": full, "positive_pairs": profile, "france_test_profile": france}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
