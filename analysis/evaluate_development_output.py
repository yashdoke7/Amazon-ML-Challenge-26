"""Score a complete development-split output, including singletons and caps."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_first_matcher import DATA, ROOT, entity_split, f05_for_query


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--matching",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    args=parser.parse_args()
    truth={}
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if entity_split(row["source1_entity_id"])=="development":
                truth[row["source1_entity_id"]]=set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    country={}
    with (DATA / "train_source1.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in truth:
                country[row["entity_id"]]=row["country"]
    predictions={}
    candidates={}
    for path,target,column in ((args.matching,predictions,"matched_entity_ids"),
                               (args.candidate,candidates,"candidate_entity_ids")):
        with path.open(encoding="utf-8",newline="") as stream:
            for row in csv.DictReader(stream,delimiter="\t"):
                s1=row["source1_entity_id"]
                assert s1 in truth and s1 not in target
                target[s1]=set(row[column].split(",")) if row[column] else set()
        assert set(target)==set(truth)
    assert all(predictions[s].issubset(candidates[s]) for s in truth)
    scores={s:f05_for_query(truth[s],predictions[s]) for s in truth}
    oracle={s:f05_for_query(truth[s],candidates[s]&truth[s]) for s in truth}
    by_country=defaultdict(list)
    for s,score in scores.items():
        by_country[country[s]].append(score)
    tp=sum(len(truth[s]&predictions[s]) for s in truth)
    fp=sum(len(predictions[s]-truth[s]) for s in truth)
    fn=sum(len(truth[s]-predictions[s]) for s in truth)
    output={"queries":len(truth),"candidate_pairs":sum(map(len,candidates.values())),
            "candidate_true_pairs":sum(len(truth[s]&candidates[s]) for s in truth),
            "predicted_pairs":sum(map(len,predictions.values())),
            "macro_f05":float(np.mean(list(scores.values()))),
            "oracle_macro_f05":float(np.mean(list(oracle.values()))),
            "tp":tp,"fp":fp,"fn":fn,
            "singleton_count":sum(not x for x in truth.values()),
            "singleton_accuracy":sum(not predictions[s] for s in truth if not truth[s])/sum(not x for x in truth.values()),
            "queries_over_11_predictions":sum(len(predictions[s])>11 for s in truth),
            "max_predictions":max(map(len,predictions.values())),
            "country_macro_f05":{k:float(np.mean(v)) for k,v in by_country.items()}}
    print(json.dumps(output,indent=2),flush=True)


if __name__=="__main__":
    main()
