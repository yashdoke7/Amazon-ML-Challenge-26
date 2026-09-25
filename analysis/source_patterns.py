"""Count selected observed text channels by source/country, then show examples."""
import json
from pathlib import Path
import duckdb

root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
con = duckdb.connect()
out = {}
for split in ("train", "test"):
    for source in (1, 2, 3):
        path = (root / split / f"{split}_source{source}.tsv").as_posix()
        rel = f"read_csv('{path}', delim='\t', header=true)"
        q = f"""
        SELECT country, count(*) n_rows,
          sum(CASE WHEN business_address IS NULL OR business_address='' THEN 1 ELSE 0 END) missing_address,
          sum(CASE WHEN regexp_matches(lower(business_name), '(^|[^a-z])(d[.]?b[.]?a[.]?|a[.]?k[.]?a[.]?|t/a|trading as)([^a-z]|$)') THEN 1 ELSE 0 END) alias_marker,
          sum(CASE WHEN regexp_matches(lower(business_name), '(\\.com|\\.in|\\.fr|www\\.)') THEN 1 ELSE 0 END) domain_marker,
          avg(length(business_name)) avg_name_chars,
          avg(length(business_address)) avg_address_chars
        FROM {rel} GROUP BY country ORDER BY country
        """
        rows = con.execute(q).fetchall()
        out[f"{split}_source{source}"] = rows
        print(f"{split}_source{source}", rows, flush=True)
(Path(__file__).resolve().parent / "source_patterns_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
