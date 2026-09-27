"""Pairwise rarity feature ablation using supplied target statistics only.

Fit a same-size 35-feature control and a 41-feature variant on the same
training-owned 5k-query extract. Select thresholds/blend weights on the
development split, then test once on independent frozen validation. This is
only a feasibility screen, not a production model or package.
"""

import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import core, normalize  # noqa: E402
from generalized_features import GENERALIZED_FEATURE_NAMES, generalized_pair_features  # noqa: E402
from evaluate_generalized_model import score  # noqa: E402
from calibrate_on_development import sampled_truth as development_truth  # noqa: E402
from evaluate_first_matcher import sampled_truth as validation_truth  # noqa: E402

FILES = ("full_combined_pairs.parquet", "full_development_combined_pairs.parquet",
         "full_validation_combined_pairs.parquet")
NEW_FEATURES = ["idf_name_overlap", "idf_address_overlap", "soft_idf_name",
                "soft_idf_address", "log_target_core_freq", "log_query_core_freq"]
OUTPUT = ROOT / "analysis/probe_rarity_model_results.json"
MODEL = ROOT / "analysis/probe_rarity_small_model.joblib"


@lru_cache(maxsize=800_000)
def tokens(value):
    return tuple(sorted({tok for tok in normalize(value).split() if len(tok) >= 4}))


def core_key(value):
    return "".join(c for c in core(value) if c.isalnum())


def rarity_features(qname, tname, qaddr, taddr, country, name_df, addr_df,
                    core_freq, target_core_freq, target_id):
    def values(qtext, ttext, dfs):
        qt, tt = tokens(qtext), tokens(ttext)
        if not qt or not tt:
            return 0., 0.
        qweights = {tok: math.log1p(5_000_000 / max(1, dfs.get((country, tok), 1))) for tok in qt}
        tweights = {tok: math.log1p(5_000_000 / max(1, dfs.get((country, tok), 1))) for tok in tt}
        denom = max(1., min(sum(qweights.values()), sum(tweights.values())))
        exact = sum(qweights[tok] for tok in set(qt) & set(tt)) / denom
        soft = 0.
        for qtok in qt:
            best = max(JaroWinkler.normalized_similarity(qtok, ttok) for ttok in tt)
            if best >= 0.88:
                soft += qweights[qtok] * best
        return min(1., exact), min(1., soft / denom)

    name, soft_name = values(core(qname), core(tname), name_df)
    address, soft_address = values(qaddr, taddr, addr_df)
    query_freq = core_freq.get((country, core_key(qname)), 0)
    return (name, address, soft_name, soft_address,
            math.log1p(target_core_freq.get(target_id, 0)), math.log1p(query_freq))


def load_stats(frames):
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.execute("SET threads=8")
    name_df = {(c,t): d for c,t,d in con.execute("SELECT country,tok,df FROM df_name").fetchall()}
    addr_df = {(c,t): d for c,t,d in con.execute("SELECT country,tok,df FROM df_address").fetchall()}
    target_ids = pd.DataFrame({"target_id":pd.concat([f.target_id for f in frames]).drop_duplicates()})
    con.register("wanted_targets", target_ids)
    keys = con.execute("""SELECT w.target_id,k.country,k.core_key FROM wanted_targets w
        JOIN target_keys k USING(target_id)""").df()
    assert len(keys) == len(target_ids), (len(keys),len(target_ids))
    qkeys = {(country,core_key(name)) for f in frames
             for country,name in f[["country","q_name"]].drop_duplicates().itertuples(index=False,name=None)}
    wanted = pd.DataFrame(set(zip(keys.country, keys.core_key)) | qkeys,
                          columns=["country","core_key"])
    con.unregister("wanted_targets")
    con.register("wanted_core", wanted)
    freq = con.execute("""SELECT k.country,k.core_key,count(*) df FROM target_keys k
        JOIN wanted_core w USING(country,core_key) GROUP BY k.country,k.core_key""").fetchall()
    con.close()
    core_freq = {(c,k): df for c,k,df in freq}
    target_core_freq = {tid:core_freq.get((c,k),0) for tid,c,k in keys.itertuples(index=False,name=None)}
    return name_df, addr_df, core_freq, target_core_freq


def matrix(frame, stats, label):
    started = time.perf_counter()
    name_df, addr_df, core_freq, target_core_freq = stats
    base = np.empty((len(frame),35), dtype=np.float32)
    extra = np.empty((len(frame),6), dtype=np.float32)
    for i,row in enumerate(frame.itertuples(index=False)):
        base[i] = generalized_pair_features(row.q_name,row.t_name,
                                            row.q_address,row.t_address,row.target_source)
        extra[i] = rarity_features(row.q_name,row.t_name,row.q_address,row.t_address,
                                   row.country,name_df,addr_df,core_freq,
                                   target_core_freq,row.target_id)
        if (i+1) % 100_000 == 0:
            print("FEATURES",label,i+1,"SECONDS",round(time.perf_counter()-started,1),flush=True)
    return base, extra


def choose(frame, truth, probabilities):
    grid = np.round(np.arange(0.35,0.951,0.05),2)
    sweeps = [(float(t),score(frame,truth,probabilities,{"India":float(t),"other":float(t)}))
              for t in grid]
    india = max(sweeps,key=lambda x:x[1]["countries"]["India"])[0]
    other = max(sweeps,key=lambda x:x[1]["countries"]["US"])[0]
    return {"India":india,"other":other}


def main():
    started = time.perf_counter()
    train,dev,val = [pd.read_parquet(ROOT / "analysis" / f).fillna("").reset_index(drop=True)
                     for f in FILES]
    train = train.loc[train.split == "training"].reset_index(drop=True)
    print("ROWS",[len(x) for x in (train,dev,val)],"SECONDS",round(time.perf_counter()-started,1),flush=True)
    stats = load_stats((train,dev,val))
    print("STATS",[len(x) for x in stats],"SECONDS",round(time.perf_counter()-started,1),flush=True)
    xtrain,etrain = matrix(train,stats,"train")
    xdev,edev = matrix(dev,stats,"development")
    xval,eval_ = matrix(val,stats,"validation")
    y = train.is_match.to_numpy(dtype=np.int8)
    config = dict(n_estimators=900,learning_rate=0.04,num_leaves=31,
                  min_child_samples=40,colsample_bytree=0.9,reg_lambda=2.0,
                  n_jobs=8,verbosity=-1,random_state=20260927)
    control = lgb.LGBMClassifier(**config)
    variant = lgb.LGBMClassifier(**config)
    control.fit(xtrain,y,feature_name=GENERALIZED_FEATURE_NAMES)
    variant.fit(np.hstack([xtrain,etrain]),y,
                feature_name=GENERALIZED_FEATURE_NAMES+NEW_FEATURES)
    dev_truth, val_truth = development_truth(), validation_truth()
    current=joblib.load(ROOT / "analysis/hard_negative_generalized_model.joblib")
    pdev={"current":current.predict_proba(xdev)[:,1],
          "control":control.predict_proba(xdev)[:,1],
          "rarity":variant.predict_proba(np.hstack([xdev,edev]))[:,1]}
    pval={"current":current.predict_proba(xval)[:,1],
          "control":control.predict_proba(xval)[:,1],
          "rarity":variant.predict_proba(np.hstack([xval,eval_]))[:,1]}
    settings={}
    results={}
    for kind in ("current","control","rarity"):
        settings[kind]=choose(dev,dev_truth,pdev[kind])
        results[kind]={"chosen":settings[kind],
                       "development":score(dev,dev_truth,pdev[kind],settings[kind]),
                       "validation":score(val,val_truth,pval[kind],settings[kind])}
    for weight in (0.1,0.25,0.5):
        for model_name in ("control","rarity"):
            kind=f"blend_{model_name}_{weight}"
            d=(1-weight)*pdev["current"]+weight*pdev[model_name]
            v=(1-weight)*pval["current"]+weight*pval[model_name]
            selected=choose(dev,dev_truth,d)
            results[kind]={"chosen":selected,"development":score(dev,dev_truth,d,selected),
                           "validation":score(val,val_truth,v,selected)}
    report={"training_pairs":len(train),"training_positive":int(y.sum()),
            "new_feature_importance":dict(zip(NEW_FEATURES,variant.feature_importances_[-6:].tolist())),
            "results":results,"seconds":round(time.perf_counter()-started,1)}
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    joblib.dump(variant,MODEL)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
