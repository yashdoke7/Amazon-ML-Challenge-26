"""Measure whether an evidence-based per-query top-k cap helps macro F0.5."""

import csv
import json
import random
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from train_first_matcher import DATA, FEATURE_NAMES, ROOT, entity_split, f05_for_query, pair_features

N = 2000
CAPS = (None,5,8,10,11,12,15,20,30,50)


def truth_sample():
    rng=random.Random(20260925)
    sample=[]
    seen=0
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if entity_split(row["source1_entity_id"])!="validation":
                continue
            truth=set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
            item=(row["source1_entity_id"],truth)
            if seen<N:
                sample.append(item)
            else:
                j=rng.randrange(seen+1)
                if j<N:
                    sample[j]=item
            seen+=1
    return dict(sample)


def main():
    tic=time.perf_counter()
    truth=truth_sample()
    df=pd.read_parquet(ROOT / "analysis" / "full_validation_combined_pairs.parquet")
    features=np.empty((len(df),len(FEATURE_NAMES)),dtype=np.float32)
    for i,row in enumerate(df.itertuples(index=False)):
        features[i]=pair_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source)
    model=joblib.load(ROOT / "analysis" / "first_matcher_model.joblib")
    probabilities=model.predict_proba(features)[:,1]
    selected=defaultdict(list)
    for s1,target,prob in zip(df.s1_id,df.target_id,probabilities):
        if prob>=0.65:
            selected[s1].append((float(prob),target))
    for values in selected.values():
        values.sort(key=lambda p:(-p[0],p[1]))
    out={"validation_queries":len(truth),"threshold":0.65,"candidate_pairs":len(df),"selected_query_count":len(selected),"max_selected":max(map(len,selected.values())),
         "queries_over_11":sum(len(x)>11 for x in selected.values()),"caps":{}}
    for cap in CAPS:
        predicted={s1:set(target for _,target in (selected[s1] if cap is None else selected[s1][:cap])) for s1 in truth}
        tp=sum(len(truth[s1]&predicted[s1]) for s1 in truth)
        fp=sum(len(predicted[s1]-truth[s1]) for s1 in truth)
        fn=sum(len(truth[s1]-predicted[s1]) for s1 in truth)
        scores=[f05_for_query(truth[s1],predicted[s1]) for s1 in truth]
        out["caps"][str(cap)]={"macro_f05":float(np.mean(scores)),"tp":tp,"fp":fp,"fn":fn,
                                "affected_queries":sum(len(selected[s1])>(cap if cap is not None else 10**9) for s1 in truth)}
        print("cap",cap,out["caps"][str(cap)],flush=True)
    out["seconds"]=round(time.perf_counter()-tic,1)
    (ROOT / "analysis" / "prediction_cap_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
