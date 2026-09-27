"""Measure a local multilingual bi-encoder training step without saving weights.

Uses supplied training-owned positive pairs only. This is a compute feasibility
benchmark, not a retrieval/score experiment or a submission model.
"""

import os
import time
from pathlib import Path

import duckdb
import torch
from sentence_transformers import SentenceTransformer, losses

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    con = duckdb.connect()
    rows = con.execute("""SELECT q_name,q_address,t_name,t_address
        FROM read_parquet('analysis/full_combined_pairs.parquet')
        WHERE split='training' AND is_match LIMIT 2048""").fetchall()
    con.close()
    source = [f"{q or ''} | {a or ''}" for q,a,_,_ in rows]
    target = [f"{t or ''} | {a or ''}" for _,_,t,a in rows]
    start = time.perf_counter()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                device="cuda")
    print("MODEL_LOAD_SECONDS",round(time.perf_counter()-start,2),flush=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)
    timings=[]
    for step in range(20):
        lo = step*32
        batch = [source[lo:lo+32],target[lo:lo+32]]
        features = [{k:v.to("cuda") for k,v in model.tokenize(texts).items()} for texts in batch]
        torch.cuda.synchronize()
        tic=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(features,labels=None)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-tic
        timings.append(elapsed)
        if step in (0,4,9,19):
            print("STEP",step+1,"SECONDS",round(elapsed,3),
                  "LOSS",round(float(loss.item()),4),
                  "VRAM_GB",round(torch.cuda.max_memory_allocated()/1e9,2),flush=True)
    mean=sum(timings[5:])/len(timings[5:])
    print("STEADY_SECONDS_PER_STEP",round(mean,4),
          "PROJECTED_HOURS_7P6M_PAIRS_3_EPOCHS_B64_LOWER_BOUND",
          round(mean*(7_600_000*3/64)/3600,2),flush=True)


if __name__ == "__main__":
    main()
