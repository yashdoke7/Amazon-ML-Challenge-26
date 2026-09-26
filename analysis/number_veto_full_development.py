"""Evaluate a conservative nearby-house-number veto on full development output."""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from validation import entity_split, f05_for_query

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
INPUT = ROOT / "tmp" / "development_full" / "expanded_exclusive.tsv"
OUTPUT = ROOT / "tmp" / "development_full" / "expanded_number_veto_us.tsv"
NUMBER = re.compile(r"\d+")


def first_number(value):
    found = NUMBER.search(value or "")
    return int(found.group()) if found else -1


def score(predictions, truth, countries):
    values = {s: f05_for_query(truth[s], predictions[s]) for s in truth}
    return {"macro_f05": round(float(np.mean(list(values.values()))), 6),
            "tp": sum(len(predictions[s] & truth[s]) for s in truth),
            "fp": sum(len(predictions[s] - truth[s]) for s in truth),
            "fn": sum(len(truth[s] - predictions[s]) for s in truth),
            "country_macro_f05": {country: round(float(np.mean([values[s] for s in truth if countries[s] == country])), 6)
                                    for country in sorted(set(countries.values()))}}


def main():
    predictions = {}
    with INPUT.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            predictions[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    truth = {}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            s1 = row["source1_entity_id"]
            if entity_split(s1) == "development":
                truth[s1] = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    assert set(predictions) == set(truth)
    query = {}
    with (DATA / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["entity_id"] in truth:
                query[row["entity_id"]] = row
    pairs = [(s, target) for s, ids in predictions.items() for target in ids]
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"), read_only=True)
    con.register("selected_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id"]))
    frame = con.execute("""SELECT s.s1_id, s.target_id, t.business_address
        FROM selected_pairs s JOIN target t USING(target_id)""").df()
    con.close()
    assert len(frame) == len(pairs)
    target_number = {(s, target): first_number(address) for s, target, address in
                     frame.itertuples(index=False, name=None)}
    revised = {}
    removed = defaultdict(list)
    for s, ids in predictions.items():
        q_number = first_number(query[s]["business_address"])
        anchor = query[s]["country"] == "US" and q_number >= 0 and any(
            target_number[s, target] == q_number for target in ids)
        revised[s] = set()
        for target in ids:
            t_number = target_number[s, target]
            if anchor and t_number >= 0 and 0 < abs(t_number - q_number) <= 10:
                removed[query[s]["country"]].append((s, target))
            else:
                revised[s].add(target)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s in predictions:
            writer.writerow([s, ",".join(sorted(revised[s]))])
    countries = {s: query[s]["country"] for s in query}
    out = {"input": INPUT.name, "output": OUTPUT.name,
           "removed": {country: {"total": len(values),
                                  "tp": sum(target in truth[s] for s, target in values),
                                  "fp": sum(target not in truth[s] for s, target in values)}
                       for country, values in removed.items()},
           "before": score(predictions, truth, countries),
           "after": score(revised, truth, countries)}
    (ROOT / "analysis" / "number_veto_full_development_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
