"""Select matcher thresholds on a larger development sample, without validation labels."""

import csv
import json
import os
import random
import time

import joblib
import numpy as np
import pandas as pd

from train_first_matcher import DATA, FEATURE_NAMES, ROOT, entity_split, pair_features, score_subset

N = int(os.environ.get("DEVELOPMENT_QUERY_COUNT", "2000"))
PAIRS = ROOT / "analysis" / "full_development_combined_pairs.parquet"
OUT = ROOT / "analysis" / "development_calibration_results.json"


def sampled_truth():
    rng = random.Random(20260925)
    sample = []
    eligible = 0
    with (DATA / "train_ground_truth.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if entity_split(row["source1_entity_id"]) != "development":
                continue
            item = (row["source1_entity_id"],set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set())
            if eligible < N:
                sample.append(item)
            else:
                j = rng.randrange(eligible+1)
                if j < N:
                    sample[j] = item
            eligible += 1
    return dict(sample)


def main():
    tic=time.perf_counter()
    truth=sampled_truth()
    df=pd.read_parquet(PAIRS)
    assert set(df.s1_id).issubset(truth)
    assert len(df)==len(df[["s1_id","target_id"]].drop_duplicates())
    assert np.array_equal(df.is_match.to_numpy(),np.array([target in truth[s1] for s1,target in zip(df.s1_id,df.target_id)]))
    features=np.empty((len(df),len(FEATURE_NAMES)),dtype=np.float32)
    for i,row in enumerate(df.itertuples(index=False)):
        features[i]=pair_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source)
        if i and i%100_000==0:
            print("features",i,"seconds",round(time.perf_counter()-tic,1),flush=True)
    ids=list(truth)
    thresholds=sorted(set(float(round(x,3)) for x in np.arange(0.3,0.976,0.025)) | {0.98,0.99,0.995})
    out={"development_queries":len(ids),"candidate_pairs":len(df),"candidate_true_pairs":int(df.is_match.sum()),
         "thresholds":thresholds,"models":{}}
    for label,path in (("core","first_matcher_model.joblib"),
                       ("combined","first_matcher_combined_model.joblib")):
        model=joblib.load(ROOT/"analysis"/path)
        assert list(model.feature_name_)==FEATURE_NAMES
        probabilities=model.predict_proba(features)[:,1]
        curve=[{"threshold":t,**score_subset(ids,truth,df,probabilities,t)} for t in thresholds]
        best=max(curve,key=lambda row:row["macro_f05"])
        out["models"][label]={"model_file":path,"best":best,"curve":curve}
        print(label,"best threshold",best["threshold"],"macro F0.5",best["macro_f05"],
              "singleton accuracy",best["singleton_accuracy"],flush=True)
    out["seconds"]=round(time.perf_counter()-tic,1)
    OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("wrote",OUT,"seconds",out["seconds"],flush=True)


if __name__=="__main__":
    main()
