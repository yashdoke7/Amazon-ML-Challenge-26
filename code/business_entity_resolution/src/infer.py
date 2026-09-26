"""Generate last-stage candidates and predictions from supplied TSVs only."""

import argparse
import csv
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, pair_features


LEGAL = r"(^|[^a-z])(inc|llc|ltd|limited|private|pvt|corp|corporation|llp|co|company|sas|sarl|sa|eurl)([^a-z]|$)"
CORE = rf"regexp_replace(regexp_replace(regexp_replace(lower(business_name), '{LEGAL}', ' ', 'g'), '{LEGAL}', ' ', 'g'), '[^\p{{L}}\p{{M}}\p{{N}}]+', '', 'g')"
ADDR_TOKENS = r"list_distinct(list_filter(string_split(trim(regexp_replace(lower(coalesce(business_address,'')), '[^\p{L}\p{M}\p{N}]+', ' ', 'g')), ' '), x -> length(x)>=2))"


def query_batches(path, size, wanted, maximum, skip):
    batch = []
    seen = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if wanted is not None and row["entity_id"] not in wanted:
                continue
            seen += 1
            if seen <= skip:
                continue
            batch.append(row)
            if len(batch) == size:
                yield batch
                batch = []
            if maximum and seen >= maximum:
                break
    if batch:
        yield batch


def candidates(con, rows, combined=True):
    con.register("query_frame", pd.DataFrame(rows))
    con.execute("CREATE TEMP TABLE q AS SELECT entity_id s1_id,business_name,business_address,country FROM query_frame")
    for label, field in (("name", "business_name"), ("address", "business_address")):
        norm = rf"trim(regexp_replace(lower({field}), '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))"
        con.execute(f"""
            CREATE TEMP TABLE q_{label} AS
            SELECT DISTINCT s1_id,country,tok FROM (
                SELECT s1_id,country,unnest(string_split({norm}, ' ')) tok FROM q
            ) WHERE length(tok)>=4
        """)
        con.execute(f"""
            CREATE TEMP TABLE selected_{label} AS
            SELECT s1_id,country,tok FROM (
                SELECT q.s1_id,q.country,q.tok,d.df,
                       row_number() OVER (PARTITION BY q.s1_id ORDER BY d.df,q.tok) rn
                FROM q_{label} q JOIN df_{label} d USING(country,tok)
                WHERE d.df<=3000
            ) WHERE rn<=3
        """)
    con.execute("""
        CREATE TEMP TABLE high AS
        SELECT q.s1_id,t.target_id FROM selected_name q JOIN postings_name t USING(country,tok)
        UNION
        SELECT q.s1_id,t.target_id FROM selected_address q JOIN postings_address t USING(country,tok)
    """)
    con.execute("""
        CREATE TEMP TABLE ranked AS
        WITH scored AS (
            SELECT p.s1_id,p.target_id,
                   jaro_winkler_similarity(lower(q.business_name),lower(t.business_name)) ns,
                   jaro_winkler_similarity(lower(q.business_address),lower(coalesce(t.business_address,''))) ads
            FROM high p JOIN q USING(s1_id) JOIN target t USING(target_id)
        )
        SELECT s1_id,target_id,ns,ads,
               row_number() OVER (PARTITION BY s1_id ORDER BY ns DESC,target_id) name_rank,
               row_number() OVER (PARTITION BY s1_id ORDER BY ads DESC,target_id) address_rank
        FROM scored
    """)
    con.execute(f"CREATE TEMP TABLE q_keys AS SELECT s1_id,country,{CORE} core_key,strip_accents({CORE}) accent_key FROM q")
    accent_union = """
        UNION SELECT q.s1_id,t.target_id FROM q_keys q JOIN target_keys t USING(country,accent_key)
        WHERE q.accent_key!=''
    """ if combined else ""
    con.execute(f"""
        CREATE TEMP TABLE all_pairs AS
        SELECT s1_id,target_id FROM ranked WHERE name_rank<=100 OR address_rank<=100
        UNION
        SELECT q.s1_id,t.target_id FROM q_keys q JOIN target_keys t USING(country,core_key)
        WHERE q.core_key!=''
        {accent_union}
    """)
    if combined:
        con.execute(f"CREATE TEMP TABLE q_addr_lists AS SELECT s1_id,{ADDR_TOKENS} q_tokens FROM q WHERE country='India'")
        con.execute(f"""
        CREATE TEMP TABLE overlap AS
        WITH tokenized AS (
            SELECT r.s1_id,r.target_id,r.ads,q.q_tokens,{ADDR_TOKENS} t_tokens
            FROM ranked r JOIN q_addr_lists q USING(s1_id) JOIN target t USING(target_id)
        ), scored AS (
            SELECT s1_id,target_id,ads,len(list_intersect(q_tokens,t_tokens)) n,
                   len(q_tokens) q_len,len(t_tokens) t_len FROM tokenized
        )
        SELECT s1_id,target_id FROM (
            SELECT s1_id,target_id,n,
                   row_number() OVER (PARTITION BY s1_id
                     ORDER BY n/greatest(1,least(q_len,t_len)) DESC,n DESC,ads DESC,target_id) rn
            FROM scored
        ) WHERE rn<=50 AND n>=2
        """)
    union_sql = "SELECT * FROM all_pairs UNION SELECT * FROM overlap" if combined else "SELECT * FROM all_pairs"
    frame = con.execute(f"""
        WITH union_pairs AS ({union_sql})
        SELECT p.s1_id,p.target_id,q.business_name q_name,q.business_address q_address,
               t.business_name t_name,t.business_address t_address,t.source target_source
        FROM union_pairs p JOIN q USING(s1_id) JOIN target t USING(target_id)
    """).df()
    tables = ["q", "q_name", "q_address", "selected_name", "selected_address",
              "high", "ranked", "q_keys", "all_pairs"]
    if combined:
        tables += ["q_addr_lists", "overlap"]
    for table in tables:
        con.execute(f"DROP TABLE {table}")
    con.unregister("query_frame")
    return frame


def infer(data_dir, db_path, model_path, output_dir, threshold, batch_size, wanted, maximum, resume):
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    model = joblib.load(model_path)
    assert list(model.feature_name_) == FEATURE_NAMES
    query_file = data_dir / f"{data_dir.name}_source1.tsv"
    result_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    skip = 0
    if resume:
        if not result_path.exists() or not candidate_path.exists():
            raise ValueError("Resume requires both existing output TSVs")
        def prior_ids(path):
            with path.open(encoding="utf-8") as stream:
                next(stream)
                return [line.split("\t",1)[0] for line in stream]
        prior_results = prior_ids(result_path)
        prior_candidates = prior_ids(candidate_path)
        if prior_results != prior_candidates:
            raise ValueError("Output TSV row IDs disagree; repair the partial batch before resuming")
        skip = len(prior_results)
        print("resuming after",skip,"completed queries",flush=True)
    tic = time.perf_counter()
    total_queries = skip
    total_candidates = total_matches = 0
    mode = "a" if resume else "w"
    with result_path.open(mode, encoding="utf-8", newline="") as result_stream, \
         candidate_path.open(mode, encoding="utf-8", newline="") as candidate_stream:
        result_writer = csv.writer(result_stream, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_stream, delimiter="\t", lineterminator="\n")
        if not resume:
            result_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for batch_number, rows in enumerate(query_batches(query_file,batch_size,wanted,maximum,skip), 1):
            batch_start = time.perf_counter()
            frame = candidates(con, rows)
            ids = {row["entity_id"]: [] for row in rows}
            predictions = {row["entity_id"]: [] for row in rows}
            if len(frame):
                features = np.empty((len(frame),len(FEATURE_NAMES)),dtype=np.float32)
                for i,row in enumerate(frame.itertuples(index=False)):
                    features[i] = pair_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source)
                probabilities = model.predict_proba(features)[:,1]
                for (s1,target),prob in zip(frame[["s1_id","target_id"]].itertuples(index=False,name=None),probabilities):
                    ids[s1].append(target)
                    if prob>=threshold:
                        predictions[s1].append(target)
            for row in rows:
                s1 = row["entity_id"]
                candidate_writer.writerow([s1,",".join(sorted(ids[s1]))])
                result_writer.writerow([s1,",".join(sorted(predictions[s1]))])
            candidate_stream.flush()
            result_stream.flush()
            total_queries += len(rows)
            total_candidates += len(frame)
            total_matches += sum(map(len,predictions.values()))
            print("batch",batch_number,"queries",total_queries,"candidates",total_candidates,
                  "matches",total_matches,"batch_seconds",round(time.perf_counter()-batch_start,1),
                  "total_seconds",round(time.perf_counter()-tic,1),flush=True)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    wanted = set(args.query_ids.read_text(encoding="utf-8").splitlines()) if args.query_ids else None
    infer(args.data_dir,args.db,args.model,args.output_dir,args.threshold,args.batch_size,wanted,args.max_queries,args.resume)
