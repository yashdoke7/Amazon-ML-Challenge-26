"""Audit full ground truth against source files, including country agreement."""
from pathlib import Path
import duckdb

root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
con = duckdb.connect()
con.execute("SET memory_limit='12GB'")
def rel(name):
    return f"read_csv('{(root / name).as_posix()}', delim='\t', header=true)"
q = f"""
WITH truth AS (
 SELECT g.source1_entity_id, s.country s1_country,
        unnest(string_split(g.matched_entity_ids, ',')) target_id
 FROM {rel('train_ground_truth.tsv')} g
 LEFT JOIN {rel('train_source1.tsv')} s ON g.source1_entity_id=s.entity_id
 WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != ''
), targets AS (
 SELECT entity_id, country FROM {rel('train_source2.tsv')}
 UNION ALL SELECT entity_id, country FROM {rel('train_source3.tsv')}
)
SELECT count(*) links, sum(CASE WHEN s1_country IS NULL THEN 1 ELSE 0 END) missing_s1,
       sum(CASE WHEN targets.entity_id IS NULL THEN 1 ELSE 0 END) missing_target,
       sum(CASE WHEN s1_country != targets.country THEN 1 ELSE 0 END) cross_country
FROM truth LEFT JOIN targets ON truth.target_id=targets.entity_id
"""
print(con.execute(q).fetchone())
