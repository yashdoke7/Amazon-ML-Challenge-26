"""Count exact cross-source text overlap; aggregate diagnostic only."""

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset"


def main():
    db = duckdb.connect()
    db.execute("SET threads=8")
    db.execute("SET memory_limit='8GB'")
    result = {}
    for split in ("train", "test"):
        for source in (2, 3):
            path = (DATA / split / f"{split}_source{source}.tsv").as_posix()
            db.execute(
                f"CREATE TEMP TABLE s{source} AS SELECT entity_id,business_name,"
                f"business_address,country FROM read_csv('{path}', delim='\\t', header=true)"
            )
        for kind, condition in (
            ("name_address", "a.business_name=b.business_name AND a.business_address=b.business_address"),
            ("name", "a.business_name=b.business_name"),
            ("address", "a.business_address=b.business_address AND a.business_address<>''"),
        ):
            result[f"{split}:{kind}"] = dict(db.execute(
                f"SELECT a.country,count(*) FROM s2 a WHERE EXISTS (SELECT 1 FROM s3 b "
                f"WHERE a.country=b.country AND {condition}) GROUP BY a.country"
            ).fetchall())
        db.execute("DROP TABLE s2")
        db.execute("DROP TABLE s3")
    output = ROOT / "analysis/fresh_cross_source_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
