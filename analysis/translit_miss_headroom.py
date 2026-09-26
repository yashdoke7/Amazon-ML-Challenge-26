"""Bound the possible gain from an Indian-script transliteration retrieval route."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import normalize  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"
LEGAL = {"private", "limited", "pvt", "ltd", "llp", "inc", "company", "corp", "the"}


def has_indic(text):
    return any(0x0900 <= ord(c) <= 0x0D7F for c in text or "")


def tokens(text):
    return {token for token in normalize(text).split()
            if len(token) >= 4 and token not in LEGAL}


def main():
    candidates = {}
    with (DEV / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            candidates[row["source1_entity_id"]] = set(row["candidate_entity_ids"].split(",")) \
                if row["candidate_entity_ids"] else set()
    query = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in candidates:
                query[row["entity_id"]] = row
    missed = []
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in candidates:
                for target in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []:
                    if target not in candidates[q]:
                        missed.append((q, target))
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"), read_only=True)
    con.register("missed", pd.DataFrame(missed, columns=["s1_id", "target_id"]))
    frame = con.execute("""SELECT m.s1_id,m.target_id,t.business_name,t.business_address,t.source
        FROM missed m JOIN target t USING(target_id)""").df().fillna("")
    con.close()
    assert len(frame) == len(missed)
    counts = Counter()
    samples = []
    for row in frame.itertuples(index=False):
        q = query[row.s1_id]
        country = q["country"]
        name = row.business_name
        qname = q["business_name"]
        counts[country, "all_missed"] += 1
        if not has_indic(name):
            continue
        counts[country, "indic_target"] += 1
        before = fuzz.ratio(normalize(qname), normalize(name))
        after = fuzz.ratio(normalize(qname), normalize(anyascii(name)))
        shared = tokens(qname) & tokens(anyascii(name))
        for threshold in (60, 70, 80):
            counts[country, f"translit_ratio_ge_{threshold}"] += after >= threshold
        counts[country, "shared_nonlegal_token"] += bool(shared)
        counts[country, "shared_token_and_ratio_ge_60"] += bool(shared) and after >= 60
        counts[country, "name_similarity_gain_ge_30"] += after - before >= 30
        if len(samples) < 20 and shared:
            samples.append({"s1_id": row.s1_id, "target_id": row.target_id,
                            "q_name": qname, "t_name": name,
                            "transliterated": anyascii(name),
                            "shared": sorted(shared),
                            "before": round(before, 1), "after": round(after, 1)})
    result = {"missed_links": len(missed),
              "counts": {country: {kind: value for (region, kind), value in counts.items()
                                   if region == country}
                         for country in sorted(set(q["country"] for q in query.values()))},
              "illustrative_examples": samples}
    (ROOT / "analysis" / "translit_miss_headroom_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "illustrative_examples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
