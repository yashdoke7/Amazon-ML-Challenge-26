"""Cap unusually large predicted groups using the same model's pair scores."""

import argparse
import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, pair_features


def cap_results(data_dir, db_path, model_path, input_path, output_path, cap):
    if input_path.resolve()==output_path.resolve():
        raise ValueError("Input and output paths must differ")
    tic=time.perf_counter()
    selected=[]
    affected=set()
    with input_path.open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            ids=row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            if len(ids)>cap:
                s1=row["source1_entity_id"]
                affected.add(s1)
                selected.extend((s1,target) for target in ids)
    print("affected queries",len(affected),"selected pairs",len(selected),flush=True)
    if not affected:
        shutil.copyfile(input_path,output_path)
        return
    queries={}
    with (data_dir / f"{data_dir.name}_source1.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in affected:
                queries[row["entity_id"]]=row
    assert len(queries)==len(affected)
    con=duckdb.connect(str(db_path),read_only=True)
    con.register("selected",pd.DataFrame(selected,columns=["s1_id","target_id"]))
    frame=con.execute("""
        SELECT p.s1_id,p.target_id,t.business_name,t.business_address,t.source
        FROM selected p JOIN target t USING(target_id)
    """).df()
    con.close()
    assert len(frame)==len(selected)
    features=np.empty((len(frame),len(FEATURE_NAMES)),dtype=np.float32)
    for i,row in enumerate(frame.itertuples(index=False)):
        query=queries[row.s1_id]
        features[i]=pair_features(query["business_name"],row.business_name,
                                  query["business_address"],row.business_address,row.source)
        if i and i%100_000==0:
            print("scored features",i,"seconds",round(time.perf_counter()-tic,1),flush=True)
    model=joblib.load(model_path)
    assert list(model.feature_name_)==FEATURE_NAMES
    probs=model.predict_proba(features)[:,1]
    ranked=defaultdict(list)
    for (s1,target),prob in zip(frame[["s1_id","target_id"]].itertuples(index=False,name=None),probs):
        ranked[s1].append((float(prob),target))
    keep={s1:set(target for _,target in sorted(values,key=lambda x:(-x[0],x[1]))[:cap])
          for s1,values in ranked.items()}
    assert set(keep)==affected
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with input_path.open(encoding="utf-8",newline="") as source, \
         output_path.open("w",encoding="utf-8",newline="") as destination:
        writer=csv.writer(destination,delimiter="\t",lineterminator="\n")
        writer.writerow(["source1_entity_id","matched_entity_ids"])
        for row in csv.DictReader(source,delimiter="\t"):
            s1=row["source1_entity_id"]
            if s1 in keep:
                writer.writerow([s1,",".join(sorted(keep[s1]))])
            else:
                writer.writerow([s1,row["matched_entity_ids"]])
    print("wrote",output_path,"seconds",round(time.perf_counter()-tic,1),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--db",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cap",type=int,default=11)
    args=parser.parse_args()
    cap_results(args.data_dir,args.db,args.model,args.input,args.output,args.cap)
