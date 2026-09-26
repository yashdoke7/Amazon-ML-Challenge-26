"""Rebuild the bundled LightGBM matcher from the supplied training TSVs."""

import argparse
import csv
import random
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, pair_features
from infer import candidates
from validation import entity_split


def sample_truth(data_dir, count=5000):
    rng = random.Random(20260925)
    sample = []
    heldout_targets = set()
    with (data_dir / "train_ground_truth.tsv").open(encoding="utf-8",newline="") as stream:
        for i,row in enumerate(csv.DictReader(stream,delimiter="\t")):
            s1 = row["source1_entity_id"]
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            if entity_split(s1)!="training":
                heldout_targets.update(ids)
            if i<count:
                sample.append((s1,ids))
            else:
                j = rng.randrange(i+1)
                if j<count:
                    sample[j] = (s1,ids)
    return {s1:set(ids) for s1,ids in sample},heldout_targets


def fit(data_dir, db_path, output_path, query_batch_size, sample_size):
    tic = time.perf_counter()
    truth,heldout = sample_truth(data_dir,count=sample_size)
    queries = []
    with (data_dir / "train_source1.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in truth:
                queries.append(row)
    con = duckdb.connect(str(db_path),read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    frames = []
    for start in range(0,len(queries),query_batch_size):
        frame = candidates(con,queries[start:start+query_batch_size],combined=False)
        frames.append(frame)
        print("retrieved",min(start+query_batch_size,len(queries)),"queries,",len(frame),"batch pairs",flush=True)
    con.close()
    df = pd.concat(frames,ignore_index=True)
    assert len(df)==len(df[["s1_id","target_id"]].drop_duplicates())
    df["split"] = [entity_split(s1) for s1 in df.s1_id]
    df["is_match"] = [target in truth[s1] for s1,target in zip(df.s1_id,df.target_id)]
    df = df.loc[~((df.split=="training") & ~df.is_match & df.target_id.isin(heldout))].reset_index(drop=True)
    print("training pool",len(df),"pairs",int(df.is_match.sum()),"positives",flush=True)
    features = np.empty((len(df),len(FEATURE_NAMES)),dtype=np.float32)
    for i,row in enumerate(df.itertuples(index=False)):
        features[i] = pair_features(row.q_name,row.t_name,row.q_address,row.t_address,row.target_source)
        if i and i%100_000==0:
            print("features",i,"seconds",round(time.perf_counter()-tic,1),flush=True)
    labels = df.is_match.to_numpy(dtype=np.int8)
    internal_cal = np.array([int(s1.split("-")[-1])%10==0 for s1 in df.s1_id])
    fit_mask = (df.split=="training").to_numpy() & ~internal_cal
    cal_mask = ((df.split=="training").to_numpy() & internal_cal) | (df.split=="development").to_numpy()
    model = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=31,
        min_child_samples=50,colsample_bytree=0.9,reg_lambda=2.0,n_jobs=8,
        verbosity=-1,random_state=20260925)
    model.fit(features[fit_mask],labels[fit_mask],feature_name=FEATURE_NAMES,
        eval_set=[(features[cal_mask],labels[cal_mask])],eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(30,verbose=False)])
    output_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,output_path)
    print("saved",output_path,"best iteration",model.best_iteration_,
          "seconds",round(time.perf_counter()-tic,1),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--db",type=Path,required=True)
    parser.add_argument("--model-out",type=Path,required=True)
    parser.add_argument("--query-batch-size",type=int,default=1000)
    parser.add_argument("--sample-size",type=int,default=5000)
    args=parser.parse_args()
    fit(args.data_dir,args.db,args.model_out,args.query_batch_size,args.sample_size)
