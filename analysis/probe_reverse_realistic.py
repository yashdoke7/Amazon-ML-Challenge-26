"""Sample targets without labels, retrieve toward all Source 1, audit dev pairs.

The sample is selected by target ID hash before truth is read. Every same-country
training Source 1 is indexed, so the retrieval competition is realistic.
Output pair scores stay local under tmp/. Use a separate evaluator to measure
macro F0.5 after the production matcher and group postprocessing.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import core, normalize  # noqa: E402

TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DB = ROOT / ".duckdb/train_index.duckdb"


def route(field, target_rows, s1_rows, dev_ids, top_n, writer):
    started = time.perf_counter()
    name_mode = field == "business_name"
    transform = core if name_mode else normalize
    corpus = [transform(anyascii(row[field])) for row in s1_rows]
    vectorizer = TfidfVectorizer(
        analyzer="char" if name_mode else "word",
        ngram_range=(3, 4) if name_mode else (1, 2),
        min_df=2, max_df=0.8,
        max_features=120_000 if name_mode else 200_000,
        sublinear_tf=True, dtype=np.float32,
        **({} if name_mode else {"token_pattern": r"(?u)\b\w+\b"}))
    matrix = vectorizer.fit_transform(corpus)
    transpose = matrix.T.tocsr()
    del corpus, matrix
    print("INDEX", field, "shape", transpose.shape, "seconds",
          round(time.perf_counter()-started, 1), flush=True)
    kept = 0
    searched = 0
    for first in range(0, len(target_rows), 200):
        batch = target_rows[first:first+200]
        encoded = vectorizer.transform([transform(anyascii(row[field])) for row in batch])
        similarities = sp_matmul_topn(encoded, transpose, top_n=top_n,
                                        n_threads=8, sort=True)
        for i, target in enumerate(batch):
            lo, hi = similarities.indptr[i:i+2]
            for rank, (col, sim) in enumerate(zip(
                    similarities.indices[lo:hi], similarities.data[lo:hi]), 1):
                q = s1_rows[col]["entity_id"]
                if q in dev_ids:
                    writer.writerow([q, target["target_id"], field, rank,
                                     format(float(sim), ".6f")])
                    kept += 1
        searched += len(batch)
        if searched % 20_000 == 0:
            print("SEARCH", field, searched, "of", len(target_rows),
                  "dev_pairs", kept, "seconds",
                  round(time.perf_counter()-started, 1), flush=True)
    return {"field": field, "target_rows": len(target_rows),
            "dev_pairs": kept, "seconds": round(time.perf_counter()-started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="US")
    parser.add_argument("--hash-mod", type=int, default=20)
    parser.add_argument("--hash-remainder", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--field", choices=("both", "business_name", "business_address"),
                        default="both")
    args = parser.parse_args()
    if args.hash_mod < 1 or not 0 <= args.hash_remainder < args.hash_mod:
        parser.error("Invalid hash sample")
    if not 1 <= args.top_n <= 20:
        parser.error("top-n must be 1..20")
    started = time.perf_counter()
    dev_ids = set()
    with (ROOT / "tmp/development_full/tfidf_top20_final.tsv").open(
            encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            dev_ids.add(row["source1_entity_id"])
    con = duckdb.connect(str(DB), read_only=True)
    target_rows = con.execute("""SELECT target_id,business_name,
        coalesce(business_address,'') business_address,source
        FROM target WHERE country=? AND hash(target_id) % ? = ?""",
        [args.country, args.hash_mod, args.hash_remainder]).df().to_dict("records")
    con.close()
    print("SAMPLE", len(target_rows), "country", args.country,
          "hash", args.hash_remainder, "/", args.hash_mod, flush=True)
    s1_rows = []
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["country"] == args.country:
                s1_rows.append(row)
    print("S1_INDEX", len(s1_rows), "dev_ids", len(dev_ids), flush=True)
    pairs = ROOT / f"tmp/reverse_{args.country.lower()}_sample_{args.hash_remainder}_of_{args.hash_mod}_top{args.top_n}_{args.field}.tsv"
    results = []
    with pairs.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "target_id", "field", "rank", "cosine"])
        fields = ("business_name", "business_address") if args.field == "both" else (args.field,)
        for field in fields:
            rows = [r for r in target_rows if r[field]]
            result = route(field, rows, s1_rows, dev_ids, args.top_n, writer)
            results.append(result)
            print("RESULT", json.dumps(result), flush=True)
    print("COMPLETE", json.dumps({"sample": len(target_rows), "s1_index": len(s1_rows),
          "results": results, "pair_file": str(pairs),
          "seconds": round(time.perf_counter()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
