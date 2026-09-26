"""Remove a US near-number conflict when another selected link has the exact number."""

import argparse
import csv
import itertools
import re
import time
from pathlib import Path

import duckdb
import pandas as pd

NUMBER = re.compile(r"\d+")


def first_number(value):
    found = NUMBER.search(value or "")
    return int(found.group()) if found else -1


def process_batch(con, rows, writer):
    pairs = [(q["entity_id"], target) for q, row in rows for target in
             (row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else [])]
    target_number = {}
    if pairs:
        con.register("selected_pairs", pd.DataFrame(pairs, columns=["s1_id", "target_id"]))
        frame = con.execute("""SELECT p.s1_id, p.target_id, t.business_address
            FROM selected_pairs p JOIN target t USING(target_id)""").df()
        con.unregister("selected_pairs")
        assert len(frame) == len(pairs), "A predicted target ID is missing from the index"
        target_number = {(s1, target): first_number(address) for s1, target, address in
                         frame.itertuples(index=False, name=None)}
    removed = 0
    for query, row in rows:
        s1 = query["entity_id"]
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        qnum = first_number(query["business_address"])
        anchor = query["country"] == "US" and qnum >= 0 and any(
            target_number[s1, target] == qnum for target in ids)
        if anchor:
            kept = [target for target in ids if not (
                target_number[s1, target] >= 0 and
                0 < abs(target_number[s1, target] - qnum) <= 10)]
        else:
            kept = ids
        removed += len(ids) - len(kept)
        writer.writerow([s1, ",".join(kept)])
    return len(pairs), removed


def run(data_dir, db_path, input_path, output_path, batch_size, wanted):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    query_path = data_dir / f"{data_dir.name}_source1.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_queries = total_pairs = total_removed = 0
    with query_path.open(encoding="utf-8", newline="") as query_stream, \
         input_path.open(encoding="utf-8", newline="") as match_stream, \
         output_path.open("w", encoding="utf-8", newline="") as destination:
        queries = csv.DictReader(query_stream, delimiter="\t")
        if wanted is not None:
            queries = (query for query in queries if query["entity_id"] in wanted)
        matches = csv.DictReader(match_stream, delimiter="\t")
        assert matches.fieldnames == ["source1_entity_id", "matched_entity_ids"]
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        batch = []
        for query, row in itertools.zip_longest(queries, matches):
            assert query is not None and row is not None, ("row count mismatch", total_queries)
            assert query["entity_id"] == row["source1_entity_id"], ("query ID mismatch", total_queries)
            batch.append((query, row))
            if len(batch) >= batch_size:
                pairs, removed = process_batch(con, batch, writer)
                total_queries += len(batch)
                total_pairs += pairs
                total_removed += removed
                destination.flush()
                print("queries", total_queries, "selected_pairs", total_pairs,
                      "removed", total_removed, "seconds", round(time.perf_counter()-started, 1), flush=True)
                batch = []
        if batch:
            pairs, removed = process_batch(con, batch, writer)
            total_queries += len(batch)
            total_pairs += pairs
            total_removed += removed
            print("queries", total_queries, "selected_pairs", total_pairs,
                  "removed", total_removed, "seconds", round(time.perf_counter()-started, 1), flush=True)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--query-ids", type=Path)
    args = parser.parse_args()
    wanted = set(args.query_ids.read_text(encoding="utf-8").splitlines()) if args.query_ids else None
    run(args.data_dir, args.db, args.input, args.output, args.batch_size, wanted)
