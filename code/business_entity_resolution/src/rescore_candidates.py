"""Rescore a saved candidate TSV with another local model; no retrieval rerun."""

import argparse
import csv
import itertools
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, pair_features


def rescore_batch(con, model, rows, threshold, writer):
    queries={q["entity_id"]:q for q,_ in rows}
    pairs=[(q["entity_id"],target) for q,ids in rows for target in ids]
    predictions=defaultdict(list)
    if pairs:
        con.register("batch_pairs",pd.DataFrame(pairs,columns=["s1_id","target_id"]))
        frame=con.execute("""
            SELECT p.s1_id,p.target_id,t.business_name,t.business_address,t.source
            FROM batch_pairs p JOIN target t USING(target_id)
        """).df()
        con.unregister("batch_pairs")
        assert len(frame)==len(pairs),"A candidate target ID is missing from the target index"
        features=np.empty((len(frame),len(FEATURE_NAMES)),dtype=np.float32)
        for i,row in enumerate(frame.itertuples(index=False)):
            query=queries[row.s1_id]
            features[i]=pair_features(query["business_name"],row.business_name,
                                      query["business_address"],row.business_address,row.source)
        probabilities=model.predict_proba(features)[:,1]
        for (s1,target),prob in zip(frame[["s1_id","target_id"]].itertuples(index=False,name=None),probabilities):
            if prob>=threshold:
                predictions[s1].append(target)
    for query,_ in rows:
        s1=query["entity_id"]
        writer.writerow([s1,",".join(sorted(predictions[s1]))])
    return len(pairs),sum(map(len,predictions.values()))


def rescore(data_dir, db_path, model_path, candidate_path, output_path, threshold, batch_size):
    if candidate_path.resolve()==output_path.resolve():
        raise ValueError("Output path must differ from candidate input")
    con=duckdb.connect(str(db_path),read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    model=joblib.load(model_path)
    assert list(model.feature_name_)==FEATURE_NAMES
    query_path=data_dir / f"{data_dir.name}_source1.tsv"
    output_path.parent.mkdir(parents=True,exist_ok=True)
    tic=time.perf_counter()
    total_queries=total_pairs=total_matches=0
    with query_path.open(encoding="utf-8",newline="") as query_stream, \
         candidate_path.open(encoding="utf-8",newline="") as candidate_stream, \
         output_path.open("w",encoding="utf-8",newline="") as destination:
        queries=csv.DictReader(query_stream,delimiter="\t")
        candidates=csv.DictReader(candidate_stream,delimiter="\t")
        assert candidates.fieldnames==["source1_entity_id","candidate_entity_ids"]
        writer=csv.writer(destination,delimiter="\t",lineterminator="\n")
        writer.writerow(["source1_entity_id","matched_entity_ids"])
        batch=[]
        for q,c in itertools.zip_longest(queries,candidates):
            assert q is not None and c is not None,("query/candidate row count mismatch",total_queries)
            assert q["entity_id"]==c["source1_entity_id"],("query/candidate ID mismatch",total_queries)
            ids=c["candidate_entity_ids"].split(",") if c["candidate_entity_ids"] else []
            assert len(ids)==len(set(ids)),("duplicate candidate",total_queries)
            batch.append((q,ids))
            if len(batch)>=batch_size:
                n_pairs,n_matches=rescore_batch(con,model,batch,threshold,writer)
                total_queries+=len(batch)
                total_pairs+=n_pairs
                total_matches+=n_matches
                destination.flush()
                print("queries",total_queries,"candidate_pairs",total_pairs,"matches",total_matches,
                      "seconds",round(time.perf_counter()-tic,1),flush=True)
                batch=[]
        if batch:
            n_pairs,n_matches=rescore_batch(con,model,batch,threshold,writer)
            total_queries+=len(batch)
            total_pairs+=n_pairs
            total_matches+=n_matches
            print("queries",total_queries,"candidate_pairs",total_pairs,"matches",total_matches,
                  "seconds",round(time.perf_counter()-tic,1),flush=True)
    con.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--db",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threshold",type=float,default=0.65)
    parser.add_argument("--batch-size",type=int,default=5000)
    args=parser.parse_args()
    rescore(args.data_dir,args.db,args.model,args.candidate,args.output,args.threshold,args.batch_size)
