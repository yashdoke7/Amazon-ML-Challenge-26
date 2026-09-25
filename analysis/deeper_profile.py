"""Country-level label distribution and exact train/test record overlap."""
from pathlib import Path
import duckdb

root = Path(__file__).resolve().parents[1] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
def source(split):
    return f"read_csv('{(root / split / f'{split}_source1.tsv').as_posix()}', delim='\t', header=true)"
truth = f"read_csv('{(root / 'train' / 'train_ground_truth.tsv').as_posix()}', delim='\t', header=true)"
query = f"""
SELECT s.country, count(*) n,
       sum(CASE WHEN coalesce(t.matched_entity_ids, '')='' THEN 1 ELSE 0 END) singleton,
       avg(CASE WHEN coalesce(t.matched_entity_ids, '')='' THEN 0
                ELSE 1 + length(t.matched_entity_ids)-length(replace(t.matched_entity_ids, ',', '')) END) avg_matches
FROM {source('train')} s JOIN {truth} t ON s.entity_id=t.source1_entity_id
GROUP BY s.country ORDER BY s.country
"""
print("labels_by_country", con.execute(query).fetchall(), flush=True)
query = f"""
SELECT count(*) FROM (
  SELECT business_name, business_address, country FROM {source('train')}
  INTERSECT
  SELECT business_name, business_address, country FROM {source('test')}
)
"""
print("exact_S1_record_text_overlaps_train_test", con.execute(query).fetchone()[0], flush=True)
