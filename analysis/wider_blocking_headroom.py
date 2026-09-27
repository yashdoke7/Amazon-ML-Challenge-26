"""Audit what a wider word-token funnel could reach among final candidate misses.

This is an optimistic positive-only headroom audit, not a retrieval score. It
does not count the extra false candidates, top-100 ranking, or final matching.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import normalize  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEV = ROOT / "tmp" / "development_full"
DB = ROOT / ".duckdb" / "train_index.duckdb"


def ids(path, field, top=None):
    with path.open(encoding="utf-8", newline="") as stream:
        return {r["source1_entity_id"]: set(r[field].split(",")[:top]) if r[field] else set()
                for r in csv.DictReader(stream, delimiter="\t")}


def tokens(value, minimum=2):
    return {t for t in normalize(value).split() if len(t) >= minimum}


def main():
    base = ids(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    extra = ids(ROOT / "analysis" / "india_address_tfidf_dev_candidates.tsv",
                "candidate_entity_ids", 20)
    misses = []
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in base and row["matched_entity_ids"]:
                pool = base[q] | extra.get(q, set())
                misses.extend((q, t) for t in row["matched_entity_ids"].split(",")
                              if t not in pool)
    assert len(misses) == 4391
    wanted_q = {q for q, _ in misses}
    queries = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in wanted_q:
                queries[row["entity_id"]] = row
    con = duckdb.connect(str(DB), read_only=True)
    con.register("wanted", pd.DataFrame({"target_id": sorted({t for _, t in misses})}))
    target = con.execute("SELECT t.target_id,t.business_name,t.business_address "
                         "FROM wanted w JOIN target t USING(target_id)").df().set_index("target_id")
    assert len(target) == len({t for _, t in misses})
    qt = []
    for q in queries.values():
        for field, column in (("name", "business_name"), ("address", "business_address")):
            qt.extend((q["country"], field, tok) for tok in tokens(q[column], 4))
    toks = pd.DataFrame(qt, columns=["country", "field", "tok"]).drop_duplicates()
    dfs = {}
    for field in ("name", "address"):
        con.register("wanted_tokens", toks.loc[toks.field == field, ["country", "tok"]])
        data = con.execute(f"SELECT d.country,d.tok,d.df FROM wanted_tokens w "
                           f"JOIN df_{field} d USING(country,tok)").fetchall()
        con.unregister("wanted_tokens")
        dfs[field] = {(country, tok): df for country, tok, df in data}
    con.close()
    counters = Counter()
    examples = Counter()
    stop = {"inc", "llc", "ltd", "co", "pvt", "the", "and", "for", "sas", "sarl"}
    for qid, tid in misses:
        q = queries[qid]
        t = target.loc[tid]
        shared_short = False
        shared_long = False
        shared_common_long = False
        shared_eligible = False
        shared_selected = False
        for field, column in (("name", "business_name"), ("address", "business_address")):
            qtok = tokens(q[column])
            ttok = tokens(t[column])
            common = qtok & ttok
            shared_short |= any(2 <= len(x) <= 3 and x not in stop for x in common)
            shared_long |= any(len(x) >= 4 for x in common)
            shared_common_long |= any(len(x) >= 4 and
                                      dfs[field].get((q["country"], x), 0) > 3000
                                      for x in common)
            eligible = {x for x in qtok if len(x) >= 4 and
                        0 < dfs[field].get((q["country"], x), 0) <= 3000}
            shared_eligible |= bool(common & eligible)
            top3 = set(sorted(eligible,
                              key=lambda x: (dfs[field][(q["country"], x)], x))[:3])
            shared_selected |= bool(common & top3)
        counters[(q["country"], "total")] += 1
        for label, condition in (("short_2_3_overlap", shared_short),
                                 ("long_overlap", shared_long),
                                 ("only_short_overlap", shared_short and not shared_long),
                                 ("common_long_overlap_df_gt_3000", shared_common_long),
                                 ("eligible_long_overlap", shared_eligible),
                                 ("selected_top3_overlap", shared_selected),
                                 ("no_word_overlap_len_ge_2", not shared_short and not shared_long)):
            counters[(q["country"], label)] += int(condition)
        if shared_selected:
            examples[(q["country"], "selected_top3_but_missing")] += 1
    result = {"candidate_missing_true_links": len(misses),
              "counts": {"|".join(k): v for k, v in sorted(counters.items())},
              "note": "Positive-only optimistic headroom; an overlap need not rank in top100 or pass the matcher."}
    (ROOT / "analysis" / "wider_blocking_headroom_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
