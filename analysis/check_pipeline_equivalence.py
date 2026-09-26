"""Check final SQL routes/features against frozen validation probe artifacts."""

import csv
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "code" / "business_entity_resolution" / "src"))
from features import pair_features as final_features  # noqa: E402
from infer import candidates  # noqa: E402
from train_first_matcher import pair_features as analysis_features  # noqa: E402


def main():
    combined_ref = pd.read_parquet(ROOT / "analysis" / "full_validation_combined_pairs.parquet",columns=["s1_id","target_id"])
    wanted = set(combined_ref.s1_id)
    rows = []
    path = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train" / "train_source1.tsv"
    with path.open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in wanted:
                rows.append(row)
                if len(rows)==100:
                    break
    selected = set(row["entity_id"] for row in rows)
    con = duckdb.connect(str(ROOT / ".duckdb" / "train_index.duckdb"),read_only=True)
    for combined,reference_name in ((False,"full_validation_pairs.parquet"),
                                    (True,"full_validation_combined_pairs.parquet")):
        actual = candidates(con,rows,combined=combined)
        reference = pd.read_parquet(ROOT / "analysis" / reference_name,columns=["s1_id","target_id"])
        expected_pairs = set(zip(reference.loc[reference.s1_id.isin(selected),"s1_id"],
                                 reference.loc[reference.s1_id.isin(selected),"target_id"]))
        actual_pairs = set(zip(actual.s1_id,actual.target_id))
        assert actual_pairs==expected_pairs, (reference_name,len(expected_pairs-actual_pairs),len(actual_pairs-expected_pairs))
        for row in actual.head(1000).itertuples(index=False):
            old = np.asarray(analysis_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source),dtype=np.float32)
            new = np.asarray(final_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source),dtype=np.float32)
            assert np.array_equal(old,new)
        print(reference_name,len(actual_pairs),"exact pairs, first 1000 feature vectors identical",flush=True)
    con.close()


if __name__=="__main__":
    main()
