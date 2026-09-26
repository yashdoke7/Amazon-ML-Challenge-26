"""Rescore a saved candidate TSV with another local model; no retrieval rerun."""

import argparse
import csv
import itertools
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from model_features import extractor_for


def featurize_rows(rows, model_feature_names):
    from features import FEATURE_NAMES, pair_features
    from number_features import NUMBER_FEATURE_NAMES, number_pair_features
    feature_function=pair_features if model_feature_names==FEATURE_NAMES else number_pair_features
    assert model_feature_names in (FEATURE_NAMES,NUMBER_FEATURE_NAMES)
    features=np.empty((len(rows),len(model_feature_names)),dtype=np.float32)
    for i,(q_name,t_name,q_address,t_address,source) in enumerate(rows):
        features[i]=feature_function(q_name,t_name,q_address,t_address,source)
    return features


def rescore_batch(con, model, model_feature_names, rows, threshold, writer, pool=None):
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
        feature_rows=[]
        for row in frame.itertuples(index=False):
            query=queries[row.s1_id]
            feature_rows.append((query["business_name"],row.business_name,
                                 query["business_address"],row.business_address,row.source))
        if pool is None:
            features=featurize_rows(feature_rows,model_feature_names)
        else:
            chunks=(feature_rows[i:i+50_000] for i in range(0,len(feature_rows),50_000))
            features=np.concatenate(list(pool.map(partial(featurize_rows,model_feature_names=model_feature_names),chunks)))
        probabilities=model.predict_proba(features)[:,1]
        for (s1,target),prob in zip(frame[["s1_id","target_id"]].itertuples(index=False,name=None),probabilities):
            if prob>=threshold:
                predictions[s1].append(target)
    for query,_ in rows:
        s1=query["entity_id"]
        writer.writerow([s1,",".join(sorted(predictions[s1]))])
    return len(pairs),sum(map(len,predictions.values()))


def rescore(data_dir, db_path, model_path, candidate_path, output_path, threshold, batch_size, wanted, workers):
    if candidate_path.resolve()==output_path.resolve():
        raise ValueError("Output path must differ from candidate input")
    con=duckdb.connect(str(db_path),read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    model=joblib.load(model_path)
    model_feature_names,_=extractor_for(model)
    query_path=data_dir / f"{data_dir.name}_source1.tsv"
    output_path.parent.mkdir(parents=True,exist_ok=True)
    tic=time.perf_counter()
    total_queries=total_pairs=total_matches=0
    with ProcessPoolExecutor(max_workers=workers) if workers>1 else nullcontext() as pool, \
         query_path.open(encoding="utf-8",newline="") as query_stream, \
         candidate_path.open(encoding="utf-8",newline="") as candidate_stream, \
         output_path.open("w",encoding="utf-8",newline="") as destination:
        queries=csv.DictReader(query_stream,delimiter="\t")
        if wanted is not None:
            queries=(q for q in queries if q["entity_id"] in wanted)
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
                n_pairs,n_matches=rescore_batch(con,model,model_feature_names,batch,threshold,writer,pool)
                total_queries+=len(batch)
                total_pairs+=n_pairs
                total_matches+=n_matches
                destination.flush()
                print("queries",total_queries,"candidate_pairs",total_pairs,"matches",total_matches,
                      "seconds",round(time.perf_counter()-tic,1),flush=True)
                batch=[]
        if batch:
            n_pairs,n_matches=rescore_batch(con,model,model_feature_names,batch,threshold,writer,pool)
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
    parser.add_argument("--query-ids",type=Path)
    parser.add_argument("--workers",type=int,default=1,help="CPU processes for exact pair features")
    args=parser.parse_args()
    wanted=set(args.query_ids.read_text(encoding="utf-8").splitlines()) if args.query_ids else None
    rescore(args.data_dir,args.db,args.model,args.candidate,args.output,args.threshold,args.batch_size,wanted,args.workers)
