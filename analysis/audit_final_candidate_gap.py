"""Audit final development misses without changing the frozen model or split."""

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


def read_ids(path, column, ordered=False):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: (
                    row[column].split(",") if ordered else set(row[column].split(",")))
                if row[column] else ([] if ordered else set())
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    base = read_ids(DEV / "generalized_top40_candidates.tsv", "candidate_entity_ids")
    address = read_ids(ROOT / "analysis" / "india_address_tfidf_dev_candidates.tsv",
                       "candidate_entity_ids", ordered=True)
    chosen = read_ids(DEV / "tfidf_top20_final.tsv", "matched_entity_ids")
    assert set(base) == set(chosen)
    selected = set(base)
    query = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in selected:
                query[row["entity_id"]] = row
    assert set(query) == selected
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in selected:
                truth[row["source1_entity_id"]] = (
                    set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"]
                    else set())
    assert set(truth) == selected

    rows = []
    for q, true_ids in truth.items():
        pool = base[q] | set(address.get(q, ())[:20])
        for target in true_ids:
            stage = "selected" if target in chosen[q] else (
                "rejected" if target in pool else "missing")
            rows.append((q, target, stage))
    pairs = pd.DataFrame(rows, columns=["q", "target_id", "stage"])
    wanted = pairs[["target_id"]].drop_duplicates()
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"), read_only=True)
    con.register("wanted", wanted)
    targets = con.execute("""SELECT t.target_id,t.business_name,t.business_address,
        t.source,t.country FROM wanted w JOIN target t USING(target_id)""").df()
    con.close()
    assert len(targets) == len(wanted)
    joined = pairs.merge(targets, on="target_id", validate="many_to_one")
    assert len(joined) == len(pairs)

    counts = Counter()
    examples = []
    for row in joined.itertuples(index=False):
        q = query[row.q]
        qn, tn = q["business_name"], row.business_name
        qa, ta = q["business_address"], row.business_address or ""
        name_ratio = fuzz.ratio(normalize(anyascii(qn)), normalize(anyascii(tn)))
        addr_ratio = fuzz.ratio(normalize(anyascii(qa)), normalize(anyascii(ta)))
        script = "nonascii" if any(ord(c) > 127 for c in qn + tn) else "ascii"
        segment = (row.stage, q["country"], row.source)
        counts[("stage", *segment)] += 1
        counts[("script", *segment, script)] += 1
        counts[("target_blank", *segment, bool(ta))] += 1
        counts[("query_blank", *segment, bool(qa))] += 1
        counts[("name_ratio", *segment, int(name_ratio // 20) * 20)] += 1
        counts[("address_ratio", *segment, int(addr_ratio // 20) * 20)] += 1
        if row.stage != "selected" and len(examples) < 300:
            examples.append((row.stage, q["country"], row.source, round(name_ratio, 1),
                             round(addr_ratio, 1), qn, tn, qa, ta))
    summary = {"queries": len(base), "positive_edges": len(pairs),
               "stages": dict(Counter(pairs.stage)),
               "counts": {"|".join(map(str, k)): v for k, v in sorted(counts.items())}}
    output = ROOT / "analysis" / "final_candidate_gap_results.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "tmp" / "final_candidate_gap_examples.tsv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["stage", "country", "source", "name_ratio", "address_ratio",
                         "query_name", "target_name", "query_address", "target_address"])
        writer.writerows(examples)
    print(json.dumps({"queries": summary["queries"], "positive_edges": len(pairs),
                      "stages": summary["stages"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
