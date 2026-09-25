"""Probe rare-token blocking against all train S2/S3 records.

Uses 5,000 seeded S1 queries. Builds only target postings whose tokens occur in
those queries. This is a feasibility/retrieval test, not the final blocker.
"""
import csv
import json
import os
import random
import sys
import time
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd
import regex
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
QUERY_SPLIT = os.environ.get("QUERY_SPLIT", "all")
if QUERY_SPLIT not in ("all", "development"):
    raise ValueError("QUERY_SPLIT must be 'all' or 'development'")
if QUERY_SPLIT == "development":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))
    from validation import entity_split
OUT = Path(__file__).resolve().parent / ("token_retrieval_dev_results.json" if QUERY_SPLIT=="development" else "token_retrieval_results.json")
RNG = random.Random(20260925)
N = int(os.environ.get("QUERY_COUNT", "5000"))
sample = []
eligible_seen = 0
with (ROOT / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if QUERY_SPLIT == "development" and entity_split(row["source1_entity_id"]) != "development":
            continue
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        item = (row["source1_entity_id"], ids)
        if eligible_seen < N:
            sample.append(item)
        else:
            j = RNG.randrange(eligible_seen + 1)
            if j < N:
                sample[j] = item
        eligible_seen += 1
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
out = {"sample_s1": len(queries), "sample_true_edges": len(edges),
       "query_split": QUERY_SPLIT, "eligible_queries": eligible_seen, "routes": {}}
start_all = time.perf_counter()
for field in ("business_name", "business_address"):
    label = "name" if field == "business_name" else "address"
    print("building", label, "query tokens", flush=True)
    norm = rf"trim(regexp_replace(lower({field}), '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))"
    con.execute(f"""
        CREATE TEMP TABLE q_{label}_tokens AS
        SELECT DISTINCT entity_id s1_id, country, tok
        FROM (SELECT entity_id, country, unnest(string_split({norm}, ' ')) tok FROM queries)
        WHERE length(tok)>=4
    """)
    print("query tokens", con.execute(f"SELECT count(*), count(distinct tok) FROM q_{label}_tokens").fetchone(), flush=True)
    print("scanning target", label, "tokens", flush=True)
    tic = time.perf_counter()
    con.execute(f"""
        CREATE TEMP TABLE t_{label}_tokens AS
        WITH raw_tokens AS (
          SELECT entity_id target_id, country, unnest(string_split({norm}, ' ')) tok
          FROM ({targets})
        )
        SELECT DISTINCT r.target_id, r.country, r.tok
        FROM raw_tokens r
        JOIN (SELECT DISTINCT country,tok FROM q_{label}_tokens) q
          ON r.country=q.country AND r.tok=q.tok
        WHERE length(r.tok)>=4
    """)
    print("target postings", con.execute(f"SELECT count(*), count(distinct tok) FROM t_{label}_tokens").fetchone(), "seconds", round(time.perf_counter()-tic,1), flush=True)
    con.execute(f"CREATE TEMP TABLE df_{label} AS SELECT country,tok,count(*) df FROM t_{label}_tokens GROUP BY country,tok")
    for max_df, n_tokens in ((500, 1), (1000, 2), (3000, 3)):
        route = f"{label}_tok_df{max_df}_top{n_tokens}"
        tic = time.perf_counter()
        con.execute(f"""
            CREATE TEMP TABLE q_{route} AS
            SELECT s1_id,country,tok,df FROM (
              SELECT q.s1_id,q.country,q.tok,d.df,
                     row_number() OVER (PARTITION BY q.s1_id ORDER BY d.df,q.tok) rn
              FROM q_{label}_tokens q JOIN df_{label} d USING (country,tok)
              WHERE d.df<={max_df}
            ) WHERE rn<={n_tokens}
        """)
        selected = con.execute(f"SELECT count(*),count(distinct s1_id) FROM q_{route}").fetchone()
        con.execute(f"""
            CREATE TEMP TABLE pairs_{route} AS
            SELECT DISTINCT q.s1_id,t.target_id
            FROM q_{route} q JOIN t_{label}_tokens t USING (country,tok)
        """)
        n_pairs = con.execute(f"SELECT count(*) FROM pairs_{route}").fetchone()[0]
        n_true = con.execute(f"SELECT count(*) FROM pairs_{route} p JOIN truth_edges e USING (s1_id,target_id)").fetchone()[0]
        volumes = con.execute(f"SELECT count(*) n FROM pairs_{route} GROUP BY s1_id ORDER BY n DESC LIMIT 1").fetchone()
        out["routes"][route] = {"selected_query_tokens": selected[0], "queries_with_key": selected[1],
                                "candidate_pairs": n_pairs, "true_pairs": n_true,
                                "edge_recall": n_true/len(edges), "candidate_precision": n_true/n_pairs if n_pairs else 0,
                                "max_query_candidates": volumes[0] if volumes else 0,
                                "seconds": round(time.perf_counter()-tic,1)}
        print(route, out["routes"][route], flush=True)
    con.execute(f"DROP TABLE t_{label}_tokens")
    con.execute(f"DROP TABLE df_{label}")

combinations = {
    "token_low_union": ["name_tok_df500_top1", "address_tok_df500_top1"],
    "token_mid_union": ["name_tok_df1000_top2", "address_tok_df1000_top2"],
    "token_high_union": ["name_tok_df3000_top3", "address_tok_df3000_top3"],
}
for route, members in combinations.items():
    tic = time.perf_counter()
    con.execute(f"CREATE TEMP TABLE pairs_{route} AS " + " UNION ".join(f"SELECT * FROM pairs_{m}" for m in members))
    n_pairs = con.execute(f"SELECT count(*) FROM pairs_{route}").fetchone()[0]
    n_true = con.execute(f"SELECT count(*) FROM pairs_{route} p JOIN truth_edges e USING (s1_id,target_id)").fetchone()[0]
    out["routes"][route] = {"candidate_pairs": n_pairs, "true_pairs": n_true,
                            "edge_recall": n_true/len(edges), "candidate_precision": n_true/n_pairs,
                            "seconds": round(time.perf_counter()-tic,1)}
    print(route, out["routes"][route], flush=True)

# Slice analysis uses only the 17k labeled pairs, never the full candidate text.
truth_rows = con.execute(f"""
    SELECT e.s1_id,e.target_id,q.country,q.business_name,q.business_address,
           t.business_name,t.business_address
    FROM truth_edges e JOIN queries q ON e.s1_id=q.entity_id
    JOIN ({targets}) t ON e.target_id=t.entity_id
""").fetchall()
token_rx = regex.compile(r"[\p{L}\p{M}\p{N}]+")
nonlatin_rx = regex.compile(r"[\p{Devanagari}\p{Tamil}\p{Kannada}\p{Telugu}\p{Bengali}\p{Gujarati}\p{Gurmukhi}\p{Malayalam}]")
def basic(text):
    return " ".join(token_rx.findall(unicodedata.normalize("NFKC", text or "").casefold()))
truth_slices = {}
for s1_id,target_id,country,s1_name,s1_addr,t_name,t_addr in truth_rows:
    name_sim = fuzz.token_sort_ratio(basic(s1_name),basic(t_name))
    addr_sim = fuzz.token_sort_ratio(basic(s1_addr),basic(t_addr)) if t_addr else 0
    truth_slices[(s1_id,target_id)] = [country, target_id[:2], f"{country}/{target_id[:2]}"] + [
        label for label, flag in (("cross_script",bool(nonlatin_rx.search(t_name or ""))),
                                  ("missing_address",not t_addr),
                                  ("weak_name",name_sim<50),
                                  ("weak_address",addr_sim<50),
                                  ("both_weak",name_sim<50 and addr_sim<50)) if flag]
for route in combinations:
    hits = set(con.execute(f"SELECT p.s1_id,p.target_id FROM pairs_{route} p JOIN truth_edges e USING (s1_id,target_id)").fetchall())
    totals = {}
    recovered = {}
    for pair,categories in truth_slices.items():
        for category in categories:
            totals[category] = totals.get(category,0)+1
            recovered[category] = recovered.get(category,0)+(pair in hits)
    out["routes"][route]["slice_recall"] = {k: {"n": totals[k],"recall": recovered[k]/totals[k]} for k in totals}
    print(route, "slices", out["routes"][route]["slice_recall"], flush=True)

# Test whether a cheap pair score can shrink the high-recall token pool. This
# score is intentionally simple and is only a pre-ranking feasibility probe.
tic = time.perf_counter()
print("scoring high token candidates", flush=True)
con.execute(f"""
    CREATE TEMP TABLE ranked_high AS
    WITH scored AS (
      SELECT p.s1_id,p.target_id,
             jaro_winkler_similarity(lower(q.business_name),lower(t.business_name)) ns,
             jaro_winkler_similarity(lower(q.business_address),lower(coalesce(t.business_address,''))) ads
      FROM pairs_token_high_union p
      JOIN queries q ON p.s1_id=q.entity_id
      JOIN ({targets}) t ON p.target_id=t.entity_id
    )
    SELECT s1_id,target_id,ns,ads,
           row_number() OVER (PARTITION BY s1_id ORDER BY ns DESC, target_id) name_rank,
           row_number() OVER (PARTITION BY s1_id ORDER BY ads DESC, target_id) address_rank,
           row_number() OVER (PARTITION BY s1_id
              ORDER BY (0.65*greatest(ns,ads)+0.35*least(ns,ads)) DESC, target_id) joint_rank
    FROM scored
""")
print("scored candidates in", round(time.perf_counter()-tic,1), "seconds", flush=True)
out["prerank"] = {}
for k in (20,50,100,200):
    n_pairs = con.execute(f"SELECT count(*) FROM ranked_high WHERE joint_rank<={k}").fetchone()[0]
    hits = set(con.execute(f"""
        SELECT r.s1_id,r.target_id FROM ranked_high r
        JOIN truth_edges e USING (s1_id,target_id) WHERE r.joint_rank<={k}
    """).fetchall())
    totals = {}
    recovered = {}
    for pair,categories in truth_slices.items():
        for category in categories:
            totals[category] = totals.get(category,0)+1
            recovered[category] = recovered.get(category,0)+(pair in hits)
    out["prerank"][f"joint_top{k}"] = {"candidate_pairs": n_pairs,"true_pairs": len(hits),
       "edge_recall": len(hits)/len(edges),"slice_recall": {cat: recovered[cat]/totals[cat] for cat in totals}}
    print("joint_top",k,"candidates",n_pairs,"true",len(hits),"recall",round(len(hits)/len(edges),4),flush=True)
for k in (20,50,100):
    condition = f"r.name_rank<={k} OR r.address_rank<={k}"
    n_pairs = con.execute(f"SELECT count(*) FROM ranked_high r WHERE {condition}").fetchone()[0]
    hits = set(con.execute(f"""
        SELECT r.s1_id,r.target_id FROM ranked_high r
        JOIN truth_edges e USING (s1_id,target_id) WHERE {condition}
    """).fetchall())
    totals = {}
    recovered = {}
    for pair,categories in truth_slices.items():
        for category in categories:
            totals[category] = totals.get(category,0)+1
            recovered[category] = recovered.get(category,0)+(pair in hits)
    out["prerank"][f"name_or_address_top{k}"] = {"candidate_pairs": n_pairs,"true_pairs": len(hits),
       "edge_recall": len(hits)/len(edges),"slice_recall": {cat: recovered[cat]/totals[cat] for cat in totals}}
    print("name_or_address_top",k,"candidates",n_pairs,"true",len(hits),"recall",round(len(hits)/len(edges),4),flush=True)
out["total_seconds"] = round(time.perf_counter()-start_all,1)
high_hits = set(con.execute("""
    SELECT p.s1_id,p.target_id FROM pairs_token_high_union p
    JOIN truth_edges e USING (s1_id,target_id)
""").fetchall())
ranked_hits = set(con.execute("""
    SELECT r.s1_id,r.target_id FROM ranked_high r
    JOIN truth_edges e USING (s1_id,target_id)
    WHERE r.name_rank<=100 OR r.address_rank<=100
""").fetchall())
def complete_set_recall(hits):
    nonempty = [(s1, ids) for s1, ids in sample if ids]
    return sum(all((s1, target) in hits for target in ids) for s1, ids in nonempty)/len(nonempty)
def oracle_macro_f05(hits):
    scores = []
    for s1, ids in sample:
        if not ids:
            scores.append(1.0)
            continue
        tp = sum((s1, target) in hits for target in ids)
        fn = len(ids)-tp
        scores.append(1.25*tp/(1.25*tp+0.25*fn) if tp else 0.0)
    return sum(scores)/len(scores)
out["routes"]["token_high_union"]["complete_set_recall_nonempty"] = complete_set_recall(high_hits)
out["prerank"]["name_or_address_top100"]["complete_set_recall_nonempty"] = complete_set_recall(ranked_hits)
out["routes"]["token_high_union"]["oracle_macro_f05_ceiling"] = oracle_macro_f05(high_hits)
out["prerank"]["name_or_address_top100"]["oracle_macro_f05_ceiling"] = oracle_macro_f05(ranked_hits)
for route, table in (("token_high_union", "pairs_token_high_union"),
                     ("name_or_address_top100", "ranked_high")):
    where = "WHERE name_rank<=100 OR address_rank<=100" if route != "token_high_union" else ""
    counts = [r[0] for r in con.execute(f"SELECT count(*) FROM {table} {where} GROUP BY s1_id").fetchall()]
    counts += [0]*(len(sample)-len(counts))
    counts.sort()
    metrics = out["routes"][route] if route == "token_high_union" else out["prerank"][route]
    metrics["candidate_count_quantiles"] = {k: counts[int(p*(len(counts)-1))]
        for k,p in (("median",.5),("p90",.9),("p99",.99),("max",1.0))}
# A cheap rescue route independent of the selected rare query tokens. This is a
# retrieval probe: legal-suffix stripping must not imply a confirmed match.
legal = r"(^|[^a-z])(inc|llc|ltd|limited|private|pvt|corp|corporation|llp|co|company|sas|sarl|sa|eurl)([^a-z]|$)"
core_expr = rf"regexp_replace(regexp_replace(regexp_replace(lower(business_name), '{legal}', ' ', 'g'), '{legal}', ' ', 'g'), '[^\p{{L}}\p{{M}}\p{{N}}]+', '', 'g')"
compact_expr = r"regexp_replace(lower(business_name), '[^\p{L}\p{M}\p{N}]+', '', 'g')"
out["rescue"] = {}
rescue_hits = {}
for label,expr in (("compact",compact_expr),("core",core_expr)):
    con.execute(f"""
        CREATE TEMP TABLE pairs_rescue_{label} AS
        WITH q AS (SELECT entity_id s1_id,country,{expr} norm_key FROM queries),
             t AS (SELECT entity_id target_id,country,{expr} norm_key FROM ({targets}))
        SELECT q.s1_id,t.target_id FROM q JOIN t USING (country,norm_key)
        WHERE q.norm_key!=''
    """)
    con.execute(f"""
        CREATE TEMP TABLE pairs_rescued_{label} AS
        SELECT s1_id,target_id FROM ranked_high WHERE name_rank<=100 OR address_rank<=100
        UNION SELECT s1_id,target_id FROM pairs_rescue_{label}
    """)
    hit_rows = set(con.execute(f"""
        SELECT p.s1_id,p.target_id FROM pairs_rescued_{label} p
        JOIN truth_edges e USING (s1_id,target_id)
    """).fetchall())
    rescue_hits[label] = hit_rows
    n = con.execute(f"SELECT count(*) FROM pairs_rescued_{label}").fetchone()[0]
    totals = {}
    recovered = {}
    for pair,categories in truth_slices.items():
        for category in categories:
            totals[category] = totals.get(category,0)+1
            recovered[category] = recovered.get(category,0)+(pair in hit_rows)
    out["rescue"][label] = {"candidate_pairs":n,"true_pairs":len(hit_rows),
        "edge_recall":len(hit_rows)/len(edges),
        "complete_set_recall_nonempty":complete_set_recall(hit_rows),
        "oracle_macro_f05_ceiling":oracle_macro_f05(hit_rows),
        "slice_recall":{cat:{"n":totals[cat],"recall":recovered[cat]/totals[cat]}
                        for cat in totals}}
    counts = [r[0] for r in con.execute(f"SELECT count(*) FROM pairs_rescued_{label} GROUP BY s1_id").fetchall()]
    counts += [0]*(len(sample)-len(counts))
    counts.sort()
    out["rescue"][label]["candidate_count_quantiles"] = {k:counts[int(p*(len(counts)-1))]
        for k,p in (("median",.5),("p90",.9),("p99",.99),("max",1.0))}
    print("rescue",label,out["rescue"][label],flush=True)
for label,condition in (("india_address_top300","q.country='India' AND r.address_rank<=300"),
                        ("india_address_top500","q.country='India' AND r.address_rank<=500"),
                        ("all_address_top300","r.address_rank<=300")):
    con.execute(f"""
        CREATE TEMP TABLE pairs_{label} AS
        SELECT s1_id,target_id FROM pairs_rescued_core
        UNION
        SELECT r.s1_id,r.target_id FROM ranked_high r
        JOIN queries q ON r.s1_id=q.entity_id
        WHERE {condition}
    """)
    hits = set(con.execute(f"""
        SELECT p.s1_id,p.target_id FROM pairs_{label} p
        JOIN truth_edges e USING (s1_id,target_id)
    """).fetchall())
    n = con.execute(f"SELECT count(*) FROM pairs_{label}").fetchone()[0]
    totals,recovered = {},{}
    for pair,categories in truth_slices.items():
        for category in categories:
            totals[category] = totals.get(category,0)+1
            recovered[category] = recovered.get(category,0)+(pair in hits)
    out["rescue"][label] = {"candidate_pairs":n,"true_pairs":len(hits),
        "edge_recall":len(hits)/len(edges),
        "complete_set_recall_nonempty":complete_set_recall(hits),
        "oracle_macro_f05_ceiling":oracle_macro_f05(hits),
        "slice_recall":{cat:{"n":totals[cat],"recall":recovered[cat]/totals[cat]}
                        for cat in totals}}
    print("rescue",label,{k:v for k,v in out["rescue"][label].items()
          if k!="slice_recall"},flush=True)
domain_expr = r"regexp_extract(lower(business_name), '([a-z0-9-]+)[.](com|in|fr|org|net|co|io)(?:[^a-z]|$)', 1)"
con.execute(f"""
    CREATE TEMP TABLE pairs_rescue_domain AS
    WITH q AS (SELECT entity_id s1_id,country,regexp_replace(lower(business_name),'[^\\p{{L}}\\p{{M}}\\p{{N}}]+','','g') norm_key FROM queries),
         t AS (SELECT entity_id target_id,country,{domain_expr} norm_key FROM ({targets}))
    SELECT q.s1_id,t.target_id FROM q JOIN t USING (country,norm_key)
    WHERE q.norm_key!='' AND t.norm_key!=''
""")
con.execute("""
    CREATE TEMP TABLE pairs_rescued_domain AS
    SELECT * FROM pairs_rescued_core UNION SELECT * FROM pairs_rescue_domain
""")
domain_hits = set(con.execute("""
    SELECT p.s1_id,p.target_id FROM pairs_rescued_domain p
    JOIN truth_edges e USING (s1_id,target_id)
""").fetchall())
domain_n = con.execute("SELECT count(*) FROM pairs_rescued_domain").fetchone()[0]
out["rescue"]["domain"] = {"candidate_pairs":domain_n,"true_pairs":len(domain_hits),
    "incremental_candidate_pairs":domain_n-out["rescue"]["core"]["candidate_pairs"],
    "incremental_true_links":len(domain_hits)-out["rescue"]["core"]["true_pairs"],
    "edge_recall":len(domain_hits)/len(edges),
    "complete_set_recall_nonempty":complete_set_recall(domain_hits),
    "oracle_macro_f05_ceiling":oracle_macro_f05(domain_hits)}
print("rescue domain",out["rescue"]["domain"],flush=True)
# Address-number blocking is a targeted rescue for script changes and shortened
# addresses. It is deliberately frequency-capped because street numbers collide.
con.execute("""
    CREATE TEMP TABLE q_num_tokens AS
    SELECT DISTINCT entity_id s1_id,country,
      regexp_replace(tok,'^0+','') tok
    FROM (SELECT entity_id,country,unnest(regexp_extract_all(lower(business_address),'[0-9]+')) tok FROM queries)
    WHERE length(tok)>=2 AND regexp_replace(tok,'^0+','')!=''
""")
con.execute(f"""
    CREATE TEMP TABLE t_num_tokens AS
    SELECT DISTINCT entity_id target_id,country,
      regexp_replace(tok,'^0+','') tok
    FROM (SELECT entity_id,country,unnest(regexp_extract_all(lower(business_address),'[0-9]+')) tok FROM ({targets}))
    WHERE length(tok)>=2 AND regexp_replace(tok,'^0+','')!=''
""")
con.execute("CREATE TEMP TABLE df_num AS SELECT country,tok,count(*) df FROM t_num_tokens GROUP BY country,tok")
out["rescue"]["address_number"] = {}
for max_df,n_tokens in ((100,1),(500,1),(500,2),(1000,2)):
    route=f"num_df{max_df}_top{n_tokens}"
    con.execute(f"""
        CREATE TEMP TABLE q_{route} AS
        SELECT s1_id,country,tok FROM (
          SELECT q.s1_id,q.country,q.tok,d.df,
                 row_number() OVER (PARTITION BY q.s1_id ORDER BY d.df,q.tok) rn
          FROM q_num_tokens q JOIN df_num d USING(country,tok) WHERE d.df<={max_df}
        ) WHERE rn<={n_tokens}
    """)
    con.execute(f"""
        CREATE TEMP TABLE pairs_{route} AS
        SELECT DISTINCT q.s1_id,t.target_id FROM q_{route} q JOIN t_num_tokens t USING(country,tok)
    """)
    con.execute(f"CREATE OR REPLACE TEMP TABLE pairs_num_rescued AS SELECT * FROM pairs_rescued_core UNION SELECT * FROM pairs_{route}")
    hits=set(con.execute("""
        SELECT p.s1_id,p.target_id FROM pairs_num_rescued p JOIN truth_edges e USING(s1_id,target_id)
    """).fetchall())
    n=con.execute("SELECT count(*) FROM pairs_num_rescued").fetchone()[0]
    out["rescue"]["address_number"][route]={
      "candidate_pairs":n,"incremental_candidate_pairs":n-out["rescue"]["core"]["candidate_pairs"],
      "true_pairs":len(hits),"incremental_true_links":len(hits)-out["rescue"]["core"]["true_pairs"],
      "edge_recall":len(hits)/len(edges),"complete_set_recall_nonempty":complete_set_recall(hits),
      "oracle_macro_f05_ceiling":oracle_macro_f05(hits)}
    print("rescue",route,out["rescue"]["address_number"][route],flush=True)
case_rows = []
for s1_id,target_id,country,s1_name,s1_addr,t_name,t_addr in truth_rows:
    pair = (s1_id,target_id)
    if pair not in high_hits or pair not in ranked_hits:
        case_rows.append({"s1_id":s1_id,"target_id":target_id,"country":country,
            "s1_name":s1_name,"s1_address":s1_addr,"target_name":t_name,
            "target_address":t_addr,"slices":truth_slices[pair],
            "miss_stage":"token_high" if pair not in high_hits else "top100_prune"})
RNG.shuffle(case_rows)
chosen = []
for stage in ("token_high","top100_prune"):
    chosen.extend([row for row in case_rows if row["miss_stage"]==stage][:50])
(OUT.parent/("token_retrieval_dev_misses_results.json" if QUERY_SPLIT=="development" else "token_retrieval_misses_results.json")).write_text(
    json.dumps(chosen,ensure_ascii=False,indent=2),encoding="utf-8")
core_misses = []
core_rescued = []
for s1_id,target_id,country,s1_name,s1_addr,t_name,t_addr in truth_rows:
    pair = (s1_id,target_id)
    if pair not in rescue_hits["core"] or pair in rescue_hits["core"] and pair not in ranked_hits:
        item = {"s1_id":s1_id,"target_id":target_id,"country":country,
                "s1_name":s1_name,"s1_address":s1_addr,"target_name":t_name,
                "target_address":t_addr,"slices":truth_slices[pair]}
        (core_misses if pair not in rescue_hits["core"] else core_rescued).append(item)
RNG.shuffle(core_misses)
RNG.shuffle(core_rescued)
(OUT.parent/("core_rescue_dev_audit_results.json" if QUERY_SPLIT=="development" else "core_rescue_audit_results.json")).write_text(
    json.dumps({"remaining_misses":core_misses[:60],
                "incrementally_rescued":core_rescued[:40]},ensure_ascii=False,indent=2),
    encoding="utf-8")
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote", OUT, "total seconds", out["total_seconds"])
