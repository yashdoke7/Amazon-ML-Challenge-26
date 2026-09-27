"""Development-selected rarity veto on the current final predictions.

This is a cheap feasibility test before adding rarity to a full matcher. It
never changes submission files. Frozen validation is evaluated only after the
development sweep chooses a rule.
"""

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import core, normalize  # noqa: E402
from validation import f05_for_query  # noqa: E402

TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"
GLOB = "*fresh_fast_charaddr_tfidf_top10_fresh_india_name_tfidf_top5_final.tsv"
OUT = ROOT / "analysis/probe_rarity_veto_results.json"


def ids(value):
    return set(filter(None, (value or "").split(",")))


def read_predictions():
    files = sorted(DEV.glob(GLOB))
    assert len(files) == 2, files
    result = {}
    for split, path in (("development", files[0]), ("validation", files[1])):
        assert ("validation" in path.name) == (split == "validation")
        with path.open(encoding="utf-8", newline="") as stream:
            result[split] = {r["source1_entity_id"]: ids(r["matched_entity_ids"])
                             for r in csv.DictReader(stream, delimiter="\t")}
    return result


def load_records(predictions):
    qids = set().union(*(p.keys() for p in predictions.values()))
    tids = set().union(*(group for p in predictions.values() for group in p.values()))
    queries = {}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in qids:
                queries[row["entity_id"]] = row
    assert set(queries) == qids
    targets = {}
    for source in (2, 3):
        with (TRAIN / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["entity_id"] in tids:
                    targets[row["entity_id"]] = row
    assert set(targets) == tids
    truth = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in qids:
                truth[row["source1_entity_id"]] = ids(row["matched_entity_ids"])
    assert set(truth) == qids
    return queries, targets, truth


def frequencies(predictions, queries, targets):
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.execute("SET threads=8")
    common = set()
    for p in predictions.values():
        for q, group in p.items():
            qtokens = set(core(queries[q]["business_name"]).split())
            for t in group:
                shared = qtokens & set(core(targets[t]["business_name"]).split())
                common.update((queries[q]["country"], tok) for tok in shared if len(tok) >= 4)
    frame = pd.DataFrame(common, columns=["country", "tok"])
    con.register("wanted_tokens", frame)
    word_rows = con.execute("""SELECT w.country,w.tok,coalesce(d.df,0) df
        FROM wanted_tokens w LEFT JOIN df_name d USING(country,tok)""").fetchall()
    word_df = {(c, t): df for c, t, df in word_rows}
    wanted_targets = pd.DataFrame({"target_id": list(targets)})
    con.unregister("wanted_tokens")
    con.register("wanted_targets", wanted_targets)
    freq_rows = con.execute("""WITH wanted AS (
        SELECT w.target_id,k.country,k.core_key FROM wanted_targets w
        JOIN target_keys k USING(target_id)
    ), freq AS (
        SELECT k.country,k.core_key,count(*) AS df FROM target_keys k
        JOIN (SELECT DISTINCT country,core_key FROM wanted) w USING(country,core_key)
        GROUP BY k.country,k.core_key
    ) SELECT w.target_id,f.df FROM wanted w JOIN freq f USING(country,core_key)""").fetchall()
    con.close()
    target_freq = dict(freq_rows)
    assert len(target_freq) == len(targets)
    return word_df, target_freq


def build_features(predictions, queries, targets, truth, word_df, target_freq):
    out = {}
    for split, p in predictions.items():
        items = {}
        for q, group in p.items():
            query = queries[q]
            qname = core(query["business_name"])
            qtokens = set(qname.split())
            qaddr = normalize(query["business_address"])
            per_group = []
            for t in group:
                target = targets[t]
                ttokens = set(core(target["business_name"]).split())
                shared = qtokens & ttokens
                name_rarity = max((math.log1p(5_000_000 / max(1, word_df.get((query["country"], tok), 0)))
                                   for tok in shared if len(tok) >= 4), default=0.0)
                address_ratio = fuzz.token_set_ratio(qaddr, normalize(target["business_address"])) / 100
                has_nonascii = any(ord(c) > 127 for c in query["business_name"] + target["business_name"])
                per_group.append((t, t in truth[q], math.log1p(target_freq[t]),
                                  name_rarity, address_ratio, has_nonascii))
            items[q] = per_group
        out[split] = items
    return out


def measure(predictions, truth, items, rule):
    min_freq, max_rarity, max_addr = rule
    base = new = 0.0
    removed_tp = removed_fp = changed = 0
    for q, group in items.items():
        pred = predictions[q]
        before = f05_for_query(truth[q], pred)
        drop = {t for t, _, freq, rarity, addr, nonascii in group
                if not nonascii and freq >= min_freq and rarity < max_rarity and addr < max_addr}
        after = f05_for_query(truth[q], pred - drop)
        base += before
        new += after
        changed += bool(drop)
        removed_tp += len(drop & truth[q])
        removed_fp += len(drop - truth[q])
    return {"base": base / len(items), "new": new / len(items),
            "gain": (new-base) / len(items), "removed_tp": removed_tp,
            "removed_fp": removed_fp, "changed_queries": changed}


def main():
    started = time.perf_counter()
    predictions = read_predictions()
    queries, targets, truth = load_records(predictions)
    print("RECORDS", len(queries), len(targets), "SECONDS", round(time.perf_counter()-started,1), flush=True)
    word_df, target_freq = frequencies(predictions, queries, targets)
    print("FREQUENCIES", len(word_df), len(target_freq), "SECONDS", round(time.perf_counter()-started,1), flush=True)
    features = build_features(predictions, queries, targets, truth, word_df, target_freq)
    print("FEATURES", {s:sum(map(len,x.values())) for s,x in features.items()},
          "SECONDS", round(time.perf_counter()-started,1), flush=True)
    rules = [(math.log1p(n), r, a)
             for n in (2,5,10,20,50,100)
             for r in (0,6,8,10,12)
             for a in (0.3,0.5,0.7,0.9)]
    dev = {str(rule): measure(predictions["development"], truth, features["development"], rule)
           for rule in rules}
    chosen = max(rules, key=lambda r: dev[str(r)]["new"])
    report = {"rule": chosen, "development": dev[str(chosen)],
              "frozen_validation": measure(predictions["validation"], truth,
                                            features["validation"], chosen),
              "top_development": sorted(dev.items(), key=lambda kv:kv[1]["new"], reverse=True)[:10],
              "seconds": round(time.perf_counter()-started,1)}
    OUT.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
