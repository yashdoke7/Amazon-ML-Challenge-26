"""Independent full-index character TF-IDF retrieval audit on supplied records.

The output is a diagnostic candidate route, not a submission artifact. No labels
enter the retriever, and the vocabulary is fit on the target side only.
"""

import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", anyascii(value or "").lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--data-dir", type=Path,
                        help="Directory containing <split>_source{1,2,3}.tsv")
    parser.add_argument("--field", choices=("business_name", "business_address"), required=True)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--term-limit", type=int, default=0,
                        help="If positive, shortlist with this many strongest query n-grams")
    parser.add_argument("--shortlist", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    data = args.data_dir or (TRAIN if args.split == "train" else TRAIN.parent / "test")
    query_path = args.query_ids or (ROOT / "tmp/development_query_ids.txt"
                                     if args.split == "train" else None)
    query_ids = (set(query_path.read_text(encoding="utf-8").splitlines())
                 if query_path else None)
    key_index = {}
    texts = []
    ids_by_key = []
    for source in (2, 3):
        with (data / f"{args.split}_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["country"] != args.country or not row[args.field]:
                    continue
                key = row[args.field]
                idx = key_index.get(key)
                if idx is None:
                    idx = len(texts)
                    key_index[key] = idx
                    texts.append(norm(key))
                    ids_by_key.append([])
                ids_by_key[idx].append(row["entity_id"])
    print("TARGET_KEYS", len(texts), "SECONDS", round(time.perf_counter()-start, 1), flush=True)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4),
                                 min_df=2, max_df=0.8, max_features=args.max_features,
                                 sublinear_tf=True, dtype=np.float32)
    corpus = vectorizer.fit_transform(texts)
    corpus_t = corpus.T.tocsr()
    print("INDEX_NNZ", corpus.nnz, "FEATURES", corpus.shape[1],
          "SECONDS", round(time.perf_counter()-start, 1), flush=True)
    del texts, key_index
    if not args.term_limit:
        del corpus
    queries = []
    with (data / f"{args.split}_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if (query_ids is None or row["entity_id"] in query_ids) and row["country"] == args.country:
                queries.append((row["entity_id"], norm(row[args.field])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for offset in range(0, len(queries), args.batch_size):
            batch = queries[offset:offset + args.batch_size]
            encoded = vectorizer.transform([text for _, text in batch])
            if args.term_limit:
                term_rows, term_cols, term_values = [], [], []
                for j in range(encoded.shape[0]):
                    lo, hi = encoded.indptr[j:j+2]
                    values = encoded.data[lo:hi]
                    cols = encoded.indices[lo:hi]
                    keep = min(args.term_limit, len(values))
                    if keep:
                        chosen = np.argpartition(values, -keep)[-keep:]
                        term_rows.extend([j] * keep)
                        term_cols.extend(cols[chosen])
                        term_values.extend(values[chosen])
                limited = csr_matrix((np.asarray(term_values, dtype=np.float32),
                                      (term_rows, term_cols)), shape=encoded.shape)
                sim = (limited @ corpus_t).tocsr()
            else:
                sim = sp_matmul_topn(encoded, corpus_t, top_n=100,
                                     n_threads=args.threads, sort=True)
            for j, (qid, _) in enumerate(batch):
                lo, hi = sim.indptr[j:j+2]
                ids = []
                indices = sim.indices[lo:hi]
                if args.term_limit:
                    values = sim.data[lo:hi]
                    if len(indices) > args.shortlist:
                        indices = indices[np.argpartition(values, -args.shortlist)[-args.shortlist:]]
                    exact = (corpus[indices] @ encoded[j].T).toarray().ravel()
                    indices = indices[np.argsort(-exact, kind="stable")[:100]]
                for idx in indices:
                    ids.extend(ids_by_key[int(idx)])
                    if len(ids) >= args.top_k:
                        break
                writer.writerow([qid, ",".join(ids[:args.top_k])])
                count += 1
            if count % 2000 < len(batch):
                print("QUERIES", count, "SECONDS", round(time.perf_counter()-start, 1), flush=True)
    print("COMPLETE", count, "SECONDS", round(time.perf_counter()-start, 1), flush=True)


if __name__ == "__main__":
    main()
