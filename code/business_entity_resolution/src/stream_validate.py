"""Validate both large output TSVs without holding all candidate IDs in RAM."""

import argparse
import csv
import itertools
from pathlib import Path


def verify(test_dir: Path, matching: Path, candidate: Path):
    with (test_dir / "test_source1.tsv").open(encoding="utf-8",newline="") as source_stream, \
         matching.open(encoding="utf-8",newline="") as match_stream, \
         candidate.open(encoding="utf-8",newline="") as candidate_stream:
        source=csv.DictReader(source_stream,delimiter="\t")
        matches=csv.DictReader(match_stream,delimiter="\t")
        candidates=csv.DictReader(candidate_stream,delimiter="\t")
        assert matches.fieldnames==["source1_entity_id","matched_entity_ids"],matches.fieldnames
        assert candidates.fieldnames==["source1_entity_id","candidate_entity_ids"],candidates.fieldnames
        count=total_candidates=total_matches=empty_candidates=empty_matches=0
        for triple in itertools.zip_longest(source,matches,candidates):
            q,m,c=triple
            assert q is not None and m is not None and c is not None,("row count mismatch",count)
            s1=q["entity_id"]
            assert m["source1_entity_id"]==s1 and c["source1_entity_id"]==s1,("row ID mismatch",count,s1)
            mids=m["matched_entity_ids"].split(",") if m["matched_entity_ids"] else []
            cids=c["candidate_entity_ids"].split(",") if c["candidate_entity_ids"] else []
            mset,cset=set(mids),set(cids)
            assert len(mset)==len(mids) and len(cset)==len(cids),("duplicate ID",count)
            assert all(x.startswith(("S2-","S3-")) for x in cset),("invalid candidate ID prefix",count)
            assert mset.issubset(cset),("match absent from candidate list",count)
            count+=1
            total_matches+=len(mids)
            total_candidates+=len(cids)
            empty_matches+=not mids
            empty_candidates+=not cids
            if count%100_000==0:
                print("checked",count,"rows",flush=True)
    print("VALID",count,"queries",total_candidates,"candidates",total_matches,"matches",
          empty_candidates,"empty candidate lists",empty_matches,"empty match lists",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--test-dir",type=Path,required=True)
    parser.add_argument("--matching",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    args=parser.parse_args()
    verify(args.test_dir,args.matching,args.candidate)
