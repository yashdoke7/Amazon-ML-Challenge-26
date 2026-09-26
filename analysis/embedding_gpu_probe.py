"""Isolated, labeled name-embedding probe; never sends challenge records to an API."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer


def scores(y, lexical, embedding):
    return {
        "pairs": len(y), "positives": int(np.sum(y)),
        "lexical_auc": round(float(roc_auc_score(y, lexical)), 5),
        "embedding_auc": round(float(roc_auc_score(y, embedding)), 5),
        "lexical_ap": round(float(average_precision_score(y, lexical)), 5),
        "embedding_ap": round(float(average_precision_score(y, embedding)), 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("analysis/full_validation_combined_pairs.parquet"))
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output", type=Path, default=Path("analysis/embedding_gpu_probe_results.json"))
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA GPU is unavailable"
    frame = pd.read_parquet(args.pairs, columns=["is_match", "q_name", "t_name", "q_address", "t_address", "country"])
    frame[["q_name", "t_name", "q_address", "t_address"]] = frame[["q_name", "t_name", "q_address", "t_address"]].fillna("")
    lexical = np.array([fuzz.ratio(a or "", b or "") for a, b in zip(frame.q_name, frame.t_name)], dtype=np.float32)
    address_lexical = np.array([fuzz.ratio(a or "", b or "") for a, b in zip(frame.q_address, frame.t_address)], dtype=np.float32)
    positive = np.flatnonzero(frame.is_match.to_numpy())
    negative = np.flatnonzero(~frame.is_match.to_numpy())
    rng = np.random.default_rng(2026)
    hard_name = negative[np.argsort(lexical[negative])[-5_000:]]
    hard_address = negative[np.argsort(address_lexical[negative])[-5_000:]]
    random_negative = rng.choice(negative, size=5_000, replace=False)
    negative_sample = np.unique(np.concatenate([hard_name, hard_address, random_negative]))
    sample = np.concatenate([positive, negative_sample])
    probe = frame.iloc[sample].reset_index(drop=True)
    names = pd.Index(pd.concat([probe.q_name, probe.t_name], ignore_index=True).fillna("").unique())
    encoder = SentenceTransformer(args.model, device="cuda")
    # Warm both execution paths, then compare the same 1,024 unique names.
    small = names[:1_024].tolist()
    encoder.encode(small[:64], batch_size=64, device="cuda", show_progress_bar=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    encoder.encode(small, batch_size=128, device="cuda", show_progress_bar=False)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    encoder.encode(small, batch_size=64, device="cpu", show_progress_bar=False)
    cpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vectors = encoder.encode(names.tolist(), batch_size=128, device="cuda", show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True)
    torch.cuda.synchronize()
    all_gpu_seconds = time.perf_counter() - start
    by_name = {name: index for index, name in enumerate(names)}
    q = np.array([by_name[x or ""] for x in probe.q_name], dtype=np.int32)
    t = np.array([by_name[x or ""] for x in probe.t_name], dtype=np.int32)
    cosine = np.sum(vectors[q] * vectors[t], axis=1)
    addresses = pd.Index(pd.concat([probe.q_address, probe.t_address], ignore_index=True).unique())
    address_vectors = encoder.encode(addresses.tolist(), batch_size=128, device="cuda", show_progress_bar=False,
                                     normalize_embeddings=True, convert_to_numpy=True)
    address_lookup = {value: index for index, value in enumerate(addresses)}
    qa = np.array([address_lookup[value] for value in probe.q_address], dtype=np.int32)
    ta = np.array([address_lookup[value] for value in probe.t_address], dtype=np.int32)
    address_cosine = np.sum(address_vectors[qa] * address_vectors[ta], axis=1)
    y = probe.is_match.to_numpy(dtype=bool)
    lex = lexical[sample]
    npos = len(positive)
    results = {
        "model": args.model, "license_checked": "Apache-2.0 model card",
        "cuda_device": torch.cuda.get_device_name(0),
        "unique_names": len(names), "embedding_dimensions": vectors.shape[1],
        "unique_addresses": len(addresses),
        "cpu_1024_seconds": round(cpu_seconds, 3), "gpu_1024_seconds": round(gpu_seconds, 3),
        "gpu_all_seconds": round(all_gpu_seconds, 3),
        "all_sample": scores(y, lex, cosine),
        "all_sample_address": scores(y, address_lexical[sample], address_cosine),
        "positive_vs_hard_name": scores(np.concatenate([np.ones(npos, dtype=bool), np.zeros(len(hard_name), dtype=bool)]),
                                        np.concatenate([lex[:npos], lexical[hard_name]]),
                                        np.concatenate([cosine[:npos], cosine[npos + np.searchsorted(negative_sample, hard_name)]])),
        "positive_weak_name_count": int(np.sum(lex[:npos] < 50)),
        "positive_weak_name_cosine_median": round(float(np.median(cosine[:npos][lex[:npos] < 50])), 4),
        "negative_hard_name_cosine_median": round(float(np.median(cosine[npos + np.searchsorted(negative_sample, hard_name)])), 4),
        "positive_vs_hard_name_address": scores(
            np.concatenate([np.ones(npos, dtype=bool), np.zeros(len(hard_name), dtype=bool)]),
            np.concatenate([address_lexical[sample][:npos], address_lexical[hard_name]]),
            np.concatenate([address_cosine[:npos], address_cosine[npos + np.searchsorted(negative_sample, hard_name)]])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
