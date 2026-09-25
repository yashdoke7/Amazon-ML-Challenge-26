"""Benchmark exact-key normalization on 5k S1 queries against all train targets.

The keys are candidate-generation probes, never automatic match decisions.
"""
import csv
import json
import random
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
OUT = Path(__file__).resolve().parent / "normalization_benchmark_results.json"
RNG = random.Random(20260925)
N = 5000
sample = []
with (ROOT / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        item = (row["source1_entity_id"], ids)
        if i < N:
            sample.append(item)
        else:
            j = RNG.randrange(i + 1)
            if j < N:
                sample[j] = item
truth = dict(sample)
queries = []
with (ROOT / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["entity_id"] in truth:
            queries.append(row)
edges = [(s1, target) for s1, ids in sample for target in ids]

con = duckdb.connect()
con.execute("SET memory_limit='10GB'")
con.register("queries", pd.DataFrame(queries))
con.register("truth_edges", pd.DataFrame(edges, columns=["s1_id", "target_id"]))
targets = " UNION ALL ".join(
    f"SELECT entity_id, business_name, business_address, country FROM read_csv('{(ROOT / f'train_source{s}.tsv').as_posix()}', delim='\t', header=true)"
    for s in (2, 3)
)
name_basic = r"trim(regexp_replace(lower(business_name), '[^\p{L}\p{M}\p{N}]+', ' ', 'g'))"
addr_basic = r"trim(regexp_replace(lower(business_address), '[^\p{L}\p{M}\p{N}]+', ' ', 'g'))"
name_compact = r"regexp_replace(lower(business_name), '[^\p{L}\p{M}\p{N}]+', '', 'g')"
legal = r"(^|[^a-z])(inc|llc|ltd|limited|private|pvt|corp|corporation|llp|co|company|sas|sarl|sa|eurl)([^a-z]|$)"
name_core_compact = rf"regexp_replace(regexp_replace(regexp_replace(lower(business_name), '{legal}', ' ', 'g'), '{legal}', ' ', 'g'), '[^\p{{L}}\p{{M}}\p{{N}}]+', '', 'g')"
domain_stem = r"regexp_extract(lower(business_name), '([a-z0-9-]+)[.](com|in|fr|org|net|co|io)(?:[^a-z]|$)', 1)"
routes = {
    "raw_name": ("lower(business_name)", "lower(business_name)", "q.norm_key != ''"),
    "basic_name": (name_basic, name_basic, "q.norm_key != ''"),
    "compact_name": (name_compact, name_compact, "q.norm_key != ''"),
    "core_compact_name": (name_core_compact, name_core_compact, "q.norm_key != ''"),
    "raw_address": ("lower(business_address)", "lower(business_address)", "q.norm_key != ''"),
    "basic_address": (addr_basic, addr_basic, "q.norm_key != ''"),
    "domain_stem": (name_compact, domain_stem, "q.norm_key != '' AND t.norm_key != ''"),
    "core_domain_stem": (name_core_compact, domain_stem, "q.norm_key != '' AND t.norm_key != ''"),
}
out = {"sample_s1": len(queries), "sample_true_edges": len(edges), "routes": {}}
for route, (query_expr, target_expr, extra) in routes.items():
    print("route", route, flush=True)
    sql = f"""
    CREATE TEMP TABLE pairs_{route} AS
    WITH q AS (SELECT entity_id s1_id, country, {query_expr} norm_key FROM queries),
         t AS (SELECT entity_id target_id, country, {target_expr} norm_key FROM ({targets}))
    SELECT q.s1_id, t.target_id
    FROM q JOIN t ON q.country=t.country AND q.norm_key=t.norm_key
    WHERE {extra}
    """
    con.execute(sql)
    count = con.execute(f"SELECT count(*) FROM pairs_{route}").fetchone()[0]
    true = con.execute(f"SELECT count(*) FROM pairs_{route} p JOIN truth_edges e USING (s1_id,target_id)").fetchone()[0]
    volumes = con.execute(f"SELECT count(*) n FROM pairs_{route} GROUP BY s1_id ORDER BY n DESC LIMIT 10").fetchall()
    out["routes"][route] = {"candidate_pairs": count, "true_pairs": true,
                            "edge_recall": true / len(edges), "candidate_precision": true / count if count else 0,
                            "max_query_counts": [x[0] for x in volumes]}
    print(route, out["routes"][route], flush=True)
combinations = {
    "basic_name_or_address": ["basic_name", "basic_address"],
    "compact_name_or_basic_address": ["compact_name", "basic_address"],
    "basic_plus_domain": ["basic_name", "basic_address", "domain_stem"],
    "core_name_or_basic_address": ["core_compact_name", "basic_address"],
    "basic_plus_core_domain": ["basic_name", "basic_address", "core_domain_stem"],
}
for label, members in combinations.items():
    union_sql = " UNION ".join(f"SELECT * FROM pairs_{member}" for member in members)
    con.execute(f"CREATE TEMP TABLE pairs_{label} AS {union_sql}")
    count = con.execute(f"SELECT count(*) FROM pairs_{label}").fetchone()[0]
    true = con.execute(f"SELECT count(*) FROM pairs_{label} p JOIN truth_edges e USING (s1_id,target_id)").fetchone()[0]
    out["routes"][label] = {"candidate_pairs": count, "true_pairs": true,
                             "edge_recall": true / len(edges), "candidate_precision": true / count}
    print(label, out["routes"][label], flush=True)
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote", OUT)
