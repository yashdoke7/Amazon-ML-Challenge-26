"""Measure rare transliterated character-fragment coverage of retrieval misses.

Only counts target document frequency and known missed positives. It does not
claim global retrieval recall or make a final prediction.
"""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import normalize  # noqa: E402
from translit_miss_headroom import has_indic  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"


def ngrams(text, n):
    return {token[i:i+n] for token in normalize(text).split()
            for i in range(max(0, len(token) - n + 1))}


def main():
    started = time.perf_counter()
    frequencies = {3: Counter(), 4: Counter()}
    indic_names = 0
    for source in (2, 3):
        with (DATA / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                name = row["business_name"]
                if row["country"] != "India" or not has_indic(name):
                    continue
                indic_names += 1
                transliterated = anyascii(name)
                for n in frequencies:
                    frequencies[n].update(ngrams(transliterated, n))
        print("scanned source", source, "indic names", indic_names,
              "seconds", round(time.perf_counter()-started, 1), flush=True)
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
            if q in candidates and query[q]["country"] == "India":
                for t in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []:
                    if t not in candidates[q]:
                        missed.append((q, t))
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"), read_only=True)
    con.register("missed", pd.DataFrame(missed, columns=["s1_id", "target_id"]))
    frame = con.execute("""SELECT m.s1_id,m.target_id,t.business_name
        FROM missed m JOIN target t USING(target_id)""").df().fillna("")
    con.close()
    counts = Counter()
    for row in frame.itertuples(index=False):
        if not has_indic(row.business_name):
            continue
        qname = query[row.s1_id]["business_name"]
        translated = anyascii(row.business_name)
        ratio = fuzz.ratio(normalize(qname), normalize(translated))
        counts["indic_missed"] += 1
        if ratio >= 60:
            counts["indic_missed_ratio_ge_60"] += 1
        for n in frequencies:
            common = ngrams(qname, n) & ngrams(translated, n)
            for df_cap in (500, 1000, 3000, 10000):
                hit = any(frequencies[n][gram] <= df_cap for gram in common)
                counts[f"n{n}_df_le_{df_cap}"] += hit
                counts[f"n{n}_df_le_{df_cap}_ratio_ge_60"] += hit and ratio >= 60
    result = {"indic_target_names": indic_names,
              "distinct_ngrams": {str(n): len(freq) for n, freq in frequencies.items()},
              "counts": counts, "seconds": round(time.perf_counter()-started, 1)}
    destination = ROOT / "analysis" / "translit_ngram_headroom_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
