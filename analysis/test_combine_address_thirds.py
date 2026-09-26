"""Check exact India order and safe truncation of a longer first shard."""

import csv
import tempfile
from pathlib import Path

from combine_address_thirds import run


def write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "source1.tsv"
        files = [root / f"shard{i}.tsv" for i in range(3)]
        output = root / "combined.tsv"
        write(source, ["entity_id", "country"], [
            ("q1", "India"), ("u1", "US"), ("q2", "India"),
            ("q3", "India"), ("f1", "France"), ("q4", "India"),
            ("q5", "India"), ("q6", "India")])
        header = ["source1_entity_id", "candidate_entity_ids"]
        write(files[0], header, [("q1", "t1"), ("q2", "t2"), ("q3", "unused")])
        write(files[1], header, [("q3", "t3"), ("q4", "")])
        write(files[2], header, [("q5", "t5"), ("q6", "t6")])
        run(source, *files, output, expected=6, first_cut=2, second_cut=4)
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert [(r["source1_entity_id"], r["candidate_entity_ids"])
                for r in rows] == [("q1", "t1"), ("q2", "t2"),
                                  ("q3", "t3"), ("q4", ""),
                                  ("q5", "t5"), ("q6", "t6")]
    print("combined-address thirds test passed")


if __name__ == "__main__":
    main()
