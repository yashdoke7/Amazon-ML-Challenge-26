"""Show concrete source-text anomalies worth handling in cleaning."""
import json
from pathlib import Path

import duckdb

root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
con = duckdb.connect()
out = {}
for source in (1, 2, 3):
    rel = f"read_csv('{(root / f'train_source{source}.tsv').as_posix()}', delim='\t', header=true)"
    filters = {
        "literal_null_address": "business_address IS NOT NULL AND regexp_matches(business_address, '(?i)\\bnull\\b')",
        "leading_symbol_name": "regexp_matches(business_name, '^[^[:alnum:]]')",
        "over_100_char_name": "length(business_name)>100",
        "over_200_char_address": "length(business_address)>200",
    }
    out[f"source{source}"] = {}
    for label, where in filters.items():
        q = f"SELECT entity_id,business_name,business_address,country FROM {rel} WHERE {where} LIMIT 10"
        rows = con.execute(q).fetchall()
        out[f"source{source}"][label] = rows
        print(f"source{source} {label} {len(rows)}", flush=True)
(Path(__file__).resolve().parent / "anomaly_eda_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
