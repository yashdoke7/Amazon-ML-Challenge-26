"""Independent, aggregate-only audit of supplied 2026 entity-resolution TSVs.

Does not use existing candidates, predictions, or challenge-solution code.
"""

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset"


def source_sql(split: str, source: int) -> str:
    path = (DATA / split / f"{split}_source{source}.tsv").as_posix()
    return (
        f"read_csv('{path}', delim='\\t', header=true, "
        "columns={'entity_id':'VARCHAR','business_name':'VARCHAR',"
        "'business_address':'VARCHAR','country':'VARCHAR'})"
    )


def main() -> None:
    db = duckdb.connect()
    db.execute("SET threads=8")
    db.execute("SET memory_limit='10GB'")
    for split in ("train", "test"):
        for source in (1, 2, 3):
            db.execute(
                f"CREATE TEMP TABLE {split}{source} AS SELECT * FROM "
                f"{source_sql(split, source)}"
            )
            print("LOADED", split, source, flush=True)
    metrics = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            key = f"{split}{source}"
            country_rows = db.execute(
                    f"SELECT country, count(*) AS n, "
                    f"count(*) FILTER (WHERE business_address IS NULL OR business_address='') AS empty_address, "
                    f"count(DISTINCT (business_name,business_address,country)) AS distinct_name_address, "
                    f"count(DISTINCT (business_name,country)) AS distinct_name, "
                    f"count(DISTINCT (business_address,country)) AS distinct_address "
                    f"FROM {key} GROUP BY country"
                ).fetchall()
            metrics[key] = {
                country: {
                    "rows": n,
                    "empty_address": empty,
                    "distinct_name_address": name_address,
                    "distinct_name": name,
                    "distinct_address": address,
                }
                for country, n, empty, name_address, name, address in country_rows
            }
    for field, condition in (
        ("name_address", "a.business_name=b.business_name AND a.business_address=b.business_address"),
        ("name", "a.business_name=b.business_name"),
        ("address", "a.business_address=b.business_address AND a.business_address<>''"),
    ):
        rows = db.execute(
            f"SELECT a.country, count(*) n FROM test1 a WHERE EXISTS ("
            f"SELECT 1 FROM train1 b WHERE a.country=b.country AND {condition}) "
            f"GROUP BY a.country"
        ).fetchall()
        metrics[f"test_s1_in_train_s1_exact_{field}"] = dict(rows)
    for source in (2, 3):
        rows = db.execute(
            f"SELECT a.country, count(*) n FROM test{source} a WHERE EXISTS ("
            f"SELECT 1 FROM train{source} b WHERE a.country=b.country "
            f"AND a.business_name=b.business_name "
            f"AND a.business_address=b.business_address) GROUP BY a.country"
        ).fetchall()
        metrics[f"test_s{source}_in_train_s{source}_exact_name_address"] = dict(rows)
    gt = (DATA / "train/train_ground_truth.tsv").as_posix()
    db.execute(
        f"CREATE TEMP TABLE gt AS SELECT * FROM read_csv('{gt}', delim='\\t', "
        "header=true, columns={'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'})"
    )
    metrics["train_group_size"] = dict(
        db.execute(
            "SELECT CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids='' THEN 0 "
            "ELSE 1+length(matched_entity_ids)-length(replace(matched_entity_ids,',','')) END k, "
            "count(*) FROM gt GROUP BY k ORDER BY k"
        ).fetchall()
    )
    out = ROOT / "analysis/fresh_structural_audit_results.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
