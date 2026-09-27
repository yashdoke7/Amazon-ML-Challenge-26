"""Test the small rarity blend on the exact current development candidate union.

Keeps blank-address specialist and cap/owner steps. The 0.9317 submission is
never modified. Blend settings originate from the older development probe;
choose among them on full development, then evaluate frozen validation once.
"""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/business_entity_resolution/src"))
from validation import f05_for_query  # noqa: E402
from probe_rarity_model import load_stats,matrix  # noqa: E402

TRAIN=ROOT/"6ab10eb3b23ba_student_resource/student_resource/dataset/train"
TMP=ROOT/"tmp"
DEV=TMP/"development_full"
GLOB="*fresh_fast_charaddr_tfidf_top10_fresh_india_name_tfidf_top5_final.tsv"
OUT=ROOT/"analysis/probe_rarity_current_pool_results.json"


def ids(text):
    return set(filter(None,(text or "").split(",")))


def read_lists(path,column):
    with path.open(encoding="utf-8",newline="") as stream:
        return {r["source1_entity_id"]:ids(r[column])
                for r in csv.DictReader(stream,delimiter="\t")}


def input_data():
    finals=sorted(DEV.glob(GLOB))
    assert len(finals)==2
    splits={}
    for split,path in (("development",finals[0]),("validation",finals[1])):
        assert ("validation" in path.name)==(split=="validation")
        base=TMP/("fresh_dev_us_char_candidate_union.tsv" if split=="development"
                  else "fresh_val_us_char_candidate_union.tsv")
        extra=TMP/("fresh_india_name_char_fast_dev.tsv" if split=="development"
                   else "fresh_india_name_char_fast_devval.tsv")
        candidates=read_lists(base,"candidate_entity_ids")
        with extra.open(encoding="utf-8",newline="") as stream:
            for row in csv.DictReader(stream,delimiter="\t"):
                q=row["source1_entity_id"]
                if q in candidates:
                    candidates[q].update(filter(None,(row["candidate_entity_ids"] or "").split(",")[:5]))
        final=read_lists(path,"matched_entity_ids")
        assert final.keys()==candidates.keys()
        assert all(final[q]<=candidates[q] for q in final)
        splits[split]=(candidates,final)
    qids=set().union(*(d.keys() for d,_ in splits.values()))
    tids=set().union(*(s for d,_ in splits.values() for s in d.values()))
    queries={}
    with (TRAIN/"train_source1.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["entity_id"] in qids: queries[row["entity_id"]]=row
    assert len(queries)==len(qids)
    targets={}
    for source in (2,3):
        with (TRAIN/f"train_source{source}.tsv").open(encoding="utf-8",newline="") as stream:
            for row in csv.DictReader(stream,delimiter="\t"):
                if row["entity_id"] in tids:
                    targets[row["entity_id"]]=(row, f"S{source}")
    assert len(targets)==len(tids)
    truth={}
    with (TRAIN/"train_ground_truth.tsv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["source1_entity_id"] in qids:
                truth[row["source1_entity_id"]]=ids(row["matched_entity_ids"])
    assert len(truth)==len(qids)
    return splits,queries,targets,truth


def frame_for(candidates,queries,targets):
    rows=[]
    for q,group in candidates.items():
        qr=queries[q]
        for t in sorted(group):
            tr,source=targets[t]
            rows.append((q,t,qr["business_name"],qr["business_address"],
                         tr["business_name"],tr["business_address"],
                         qr["country"],source))
    return pd.DataFrame.from_records(rows,columns=["s1_id","target_id","q_name",
        "q_address","t_name","t_address","country","target_source"])


def evaluate(frame,prob,blank,truth,reference,thresholds,ranking_prob=None):
    ranking_prob=prob if ranking_prob is None else ranking_prob
    selected=defaultdict(list)
    for i,(q,t,country) in enumerate(frame[["s1_id","target_id","country"]].itertuples(index=False,name=None)):
        threshold=0.8 if blank[i] else thresholds["India" if country=="India" else "other"]
        if prob[i]>=threshold:
            selected[q].append((float(ranking_prob[i]),t))
    owner={}
    for q,values in selected.items():
        if len(values)>11:
            values=sorted(values,key=lambda x:(-x[0],x[1]))[:11]
            selected[q]=values
        for score,t in values:
            old=owner.get(t)
            if old is None or score>old[0] or (score==old[0] and q<old[1]):
                owner[t]=(score,q)
    predicted={q:{t for _,t in selected[q] if owner[t][1]==q} for q in truth}
    result={"macro_f05":float(np.mean([f05_for_query(y,predicted[q]) for q,y in truth.items()])),
            "tp":sum(len(y&predicted[q]) for q,y in truth.items()),
            "fp":sum(len(predicted[q]-y) for q,y in truth.items()),
            "diff_from_current_final":sum(predicted[q]!=reference[q] for q in truth),
            "changed_queries":sum(predicted[q]!=reference[q] for q in truth)}
    return result


def main():
    start=time.perf_counter()
    splits,queries,targets,truth=input_data()
    frames={split:frame_for(candidates,queries,targets)
            for split,(candidates,_) in splits.items()}
    print("FRAMES",{k:len(v) for k,v in frames.items()},"SECONDS",round(time.perf_counter()-start,1),flush=True)
    stats=load_stats(tuple(frames.values()))
    print("STATS",[len(x) for x in stats],"SECONDS",round(time.perf_counter()-start,1),flush=True)
    current=joblib.load(ROOT/"code/business_entity_resolution/generalized_model.joblib")
    rarity=joblib.load(ROOT/"analysis/probe_rarity_small_model.joblib")
    blank_model=joblib.load(ROOT/"code/business_entity_resolution/blank_frequency_model.joblib")
    scored={}
    for split,frame in frames.items():
        base,extra=matrix(frame,stats,split)
        current_prob=current.predict_proba(base)[:,1]
        rarity_prob=rarity.predict_proba(np.hstack([base,extra]))[:,1]
        blank=frame.t_address.to_numpy()==""
        if blank.any():
            xblank=np.hstack([base[blank],extra[blank,4:5]])
            current_prob[blank]=blank_model.predict_proba(xblank)[:,1]
            rarity_prob[blank]=current_prob[blank]
        scored[split]=(current_prob,rarity_prob,blank)
        print("SCORED",split,"SECONDS",round(time.perf_counter()-start,1),flush=True)
    settings=[("current",0.,{"India":0.65,"other":0.75}),
              ("rarity_0p1_old",0.1,{"India":0.7,"other":0.75}),
              ("rarity_0p25_old",0.25,{"India":0.7,"other":0.65}),
              ("rarity_0p5_old",0.5,{"India":0.7,"other":0.6}),
              ("rarity_0p1_base",0.1,{"India":0.65,"other":0.75}),
              ("rarity_0p25_base",0.25,{"India":0.65,"other":0.75}),
              ("rarity_0p5_base",0.5,{"India":0.65,"other":0.75})]
    results={}
    for label,weight,thresholds in settings:
        result={}
        for split,frame in frames.items():
            current_prob,rarity_prob,blank=scored[split]
            prob=(1-weight)*current_prob+weight*rarity_prob
            _,reference=splits[split]
            subtruth={q:truth[q] for q in reference}
            result[split]=evaluate(frame,prob,blank,subtruth,reference,
                                   thresholds,ranking_prob=current_prob)
        results[label]=result
        print("SETTING",label,result,flush=True)
    chosen=max((label for label,_,_ in settings[1:]),
               key=lambda label:results[label]["development"]["macro_f05"])
    report={"baseline_reproduction":results["current"],"chosen_on_development":chosen,
            "chosen_result":results[chosen],"all_results":results,
            "seconds":round(time.perf_counter()-start,1)}
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("FINAL",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
