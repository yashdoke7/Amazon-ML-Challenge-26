"""Recreate the saved 2,000-query frozen validation broad candidate TSV."""

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis" / "full_validation_combined_pairs.parquet"
OUTPUT = ROOT / "tmp" / "development_full" / "validation_broad_candidates.tsv"


def main():
    pairs = pd.read_parquet(SOURCE, columns=["s1_id", "target_id", "split"])
    assert set(pairs["split"]) == {"validation"}
    grouped = pairs.groupby("s1_id", sort=False)["target_id"].agg(list)
    wanted = set((ROOT / "tmp" / "validation_query_ids.txt").read_text(
        encoding="utf-8").splitlines())
    assert set(grouped.index) <= wanted
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        source = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train" / "train_source1.tsv"
        with source.open(encoding="utf-8", newline="") as query_stream:
            for row in csv.DictReader(query_stream, delimiter="\t"):
                q = row["entity_id"]
                if q in wanted:
                    ids = grouped.get(q, [])
                    assert len(ids) == len(set(ids))
                    writer.writerow([q, ",".join(ids)])
    print(len(wanted), len(pairs), OUTPUT)


if __name__ == "__main__":
    main()
