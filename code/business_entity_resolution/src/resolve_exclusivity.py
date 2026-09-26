"""Assign each multiply predicted target to one Source 1 using model scores."""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from model_features import extractor_for


def resolve(data_dir, db_path, model_path, input_path, output_path):
    if input_path.resolve()==output_path.resolve():
        raise ValueError("Input and output paths must differ")
    tic=time.perf_counter()
    first_owner={}
    collisions=defaultdict(set)
    with input_path.open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            s1=row["source1_entity_id"]
            for target in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []:
                if target in first_owner:
                    collisions[target].add(first_owner[target])
                    collisions[target].add(s1)
                else:
                    first_owner[target]=s1
    del first_owner
    affected={s1 for owners in collisions.values() for s1 in owners}
    print("collision targets",len(collisions),"affected queries",len(affected),flush=True)
    if not collisions:
        import shutil
        shutil.copyfile(input_path,output_path)
        return
    queries={}
    with (data_dir / f"{data_dir.name}_source1.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in affected:
                queries[row["entity_id"]]=row
    assert set(queries)==affected
    pairs=[(s1,target) for target,owners in collisions.items() for s1 in owners]
    con=duckdb.connect(str(db_path),read_only=True)
    con.register("collision_pairs",pd.DataFrame(pairs,columns=["s1_id","target_id"]))
    frame=con.execute("""
        SELECT p.s1_id,p.target_id,t.business_name,t.business_address,t.source
        FROM collision_pairs p JOIN target t USING(target_id)
    """).df()
    con.close()
    assert len(frame)==len(pairs)
    model=joblib.load(model_path)
    feature_names,feature_function=extractor_for(model)
    features=np.empty((len(frame),len(feature_names)),dtype=np.float32)
    for i,row in enumerate(frame.itertuples(index=False)):
        query=queries[row.s1_id]
        features[i]=feature_function(query["business_name"],row.business_name,
                                     query["business_address"],row.business_address,row.source)
    probs=model.predict_proba(features)[:,1]
    scored=defaultdict(list)
    for (s1,target),prob in zip(frame[["s1_id","target_id"]].itertuples(index=False,name=None),probs):
        scored[target].append((float(prob),s1))
    owner={target:sorted(values,key=lambda x:(-x[0],x[1]))[0][1] for target,values in scored.items()}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    removed=0
    with input_path.open(encoding="utf-8",newline="") as source, \
         output_path.open("w",encoding="utf-8",newline="") as destination:
        writer=csv.writer(destination,delimiter="\t",lineterminator="\n")
        writer.writerow(["source1_entity_id","matched_entity_ids"])
        for row in csv.DictReader(source,delimiter="\t"):
            s1=row["source1_entity_id"]
            ids=row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            kept=[t for t in ids if t not in owner or owner[t]==s1]
            removed+=len(ids)-len(kept)
            writer.writerow([s1,",".join(kept)])
    print("removed",removed,"competing links; wrote",output_path,
          "seconds",round(time.perf_counter()-tic,1),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--db",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    resolve(args.data_dir,args.db,args.model,args.input,args.output)
