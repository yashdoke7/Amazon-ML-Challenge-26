"""Build a local DuckDB index of the supplied Source 2/3 records."""

import argparse
import time
from pathlib import Path

import duckdb


def build(data_dir: Path, db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute("CREATE TABLE IF NOT EXISTS index_meta(stage VARCHAR, value VARCHAR)")
    done = {x[0] for x in con.execute("SELECT stage FROM index_meta").fetchall()}
    targets = " UNION ALL ".join(
        f"SELECT entity_id target_id,business_name,business_address,country,'S{s}' source "
        f"FROM read_csv('{(data_dir / f'{data_dir.name}_source{s}.tsv').as_posix()}', delim='{chr(9)}', header=true)"
        for s in (2, 3)
    )
    if "target" not in done:
        tic = time.perf_counter()
        con.execute(f"CREATE TABLE target AS {targets}")
        n = con.execute("SELECT count(*) FROM target").fetchone()[0]
        con.execute("INSERT INTO index_meta VALUES ('target', ?)", [str(n)])
        print("target records", n, "seconds", round(time.perf_counter()-tic, 1), flush=True)
    for label, field in (("name", "business_name"), ("address", "business_address")):
        stage = f"postings_{label}"
        if stage in done:
            continue
        tic = time.perf_counter()
        norm = rf"trim(regexp_replace(lower({field}), '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))"
        con.execute(f"""
            CREATE TABLE raw_{label} AS
            SELECT DISTINCT target_id,country,tok FROM (
                SELECT target_id,country,unnest(string_split({norm}, ' ')) tok FROM target
            ) WHERE length(tok)>=4
        """)
        con.execute(f"CREATE TABLE df_{label} AS SELECT country,tok,count(*) df FROM raw_{label} GROUP BY country,tok")
        con.execute(f"""
            CREATE TABLE postings_{label} AS
            SELECT r.target_id,r.country,r.tok,d.df
            FROM raw_{label} r JOIN df_{label} d USING(country,tok)
            WHERE d.df<=3000
        """)
        con.execute(f"DROP TABLE raw_{label}")
        n = con.execute(f"SELECT count(*) FROM postings_{label}").fetchone()[0]
        con.execute("INSERT INTO index_meta VALUES (?, ?)", [stage, str(n)])
        print(stage, n, "seconds", round(time.perf_counter()-tic, 1), flush=True)
    legal = r"(^|[^a-z])(inc|llc|ltd|limited|private|pvt|corp|corporation|llp|co|company|sas|sarl|sa|eurl)([^a-z]|$)"
    core = rf"regexp_replace(regexp_replace(regexp_replace(lower(business_name), '{legal}', ' ', 'g'), '{legal}', ' ', 'g'), '[^\p{{L}}\p{{M}}\p{{N}}]+', '', 'g')"
    if "keys" not in done:
        tic = time.perf_counter()
        con.execute(f"CREATE TABLE target_keys AS SELECT target_id,country,{core} core_key,strip_accents({core}) accent_key FROM target")
        con.execute("INSERT INTO index_meta VALUES ('keys', '1')")
        print("target keys seconds", round(time.perf_counter()-tic, 1), flush=True)
    if "blank_target_frequency" not in done:
        tic = time.perf_counter()
        con.execute("""
            CREATE TABLE blank_target_frequency AS
            WITH blank_keys AS (
                SELECT k.target_id,k.country,k.core_key FROM target_keys k
                JOIN target t USING(target_id)
                WHERE coalesce(t.business_address,'')=''
            ), wanted AS (
                SELECT DISTINCT country,core_key FROM blank_keys
            ), frequency AS (
                SELECT k.country,k.core_key,count(*) df FROM target_keys k
                JOIN wanted w USING(country,core_key)
                GROUP BY k.country,k.core_key
            )
            SELECT b.target_id,f.df FROM blank_keys b
            JOIN frequency f USING(country,core_key)
        """)
        n = con.execute("SELECT count(*) FROM blank_target_frequency").fetchone()[0]
        con.execute("INSERT INTO index_meta VALUES ('blank_target_frequency', ?)", [str(n)])
        print("blank target frequency", n, "seconds", round(time.perf_counter()-tic, 1), flush=True)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    build(args.data_dir, args.db)
