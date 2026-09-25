"""Inspect repeated reference names, addresses, and full records."""
import json
from pathlib import Path

import duckdb

path = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train" / "train_source1.tsv"
con = duckdb.connect()
rel = f"read_csv('{path.as_posix()}', delim='\t', header=true)"
out = {}
for label, fields in (("name", "country, lower(business_name)"),
                      ("address", "country, lower(business_address)"),
                      ("name_and_address", "country, lower(business_name), lower(business_address)")):
    q = f"SELECT {fields}, count(*) n FROM {rel} GROUP BY {fields} HAVING count(*)>1 ORDER BY n DESC LIMIT 20"
    rows = con.execute(q).fetchall()
    out[label] = rows
    print(label, rows[:10], flush=True)
(Path(__file__).resolve().parent / "frequency_eda_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
