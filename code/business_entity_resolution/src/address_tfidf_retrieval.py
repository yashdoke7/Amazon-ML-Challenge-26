"""Retrieve a bounded same-country address quota from supplied target records.

The TF-IDF vocabulary is fit locally from Source 2/3 addresses. No external
business data, pretrained weights, network service, or stored vector file is
needed. Output contains only chosen-country Source 1 rows in input order.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from features import normalize


def run(data_dir, output, top_k=20, batch_size=20, query_ids=None, max_queries=None,
        country="India", query_term_limit=0, shortlist_size=500, sparse_topn=False,
        search_threads=8):
    if top_k < 1 or batch_size < 1 or search_threads < 1:
        raise ValueError("top-k, batch-size, and search-threads must be positive")
    if sparse_topn and query_term_limit:
        raise ValueError("sparse-topn and query-term-limit cannot be combined")
    started = time.perf_counter()
    address_index = {}
    addresses = []
    ids_by_address = []
    target_count = 0
    for source in (2, 3):
        path = data_dir / f"{data_dir.name}_source{source}.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                address = row["business_address"]
                if row["country"] != country or not address:
                    continue
                target_count += 1
                index = address_index.get(address)
                if index is None:
                    index = len(addresses)
                    address_index[address] = index
                    addresses.append(normalize(anyascii(address)))
                    ids_by_address.append([])
                ids_by_address[index].append(row["entity_id"])
    print("target_records", target_count, "address_keys", len(addresses),
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.8,
        max_features=200_000, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, dtype=np.float32)
    corpus = vectorizer.fit_transform(addresses)
    # sparse_dot_topn converts CSC internally; materialize CSR only once so
    # large query runs do not repeat that conversion for every batch.
    transpose = corpus.T.tocsr() if sparse_topn else corpus.T
    print("tfidf_shape", corpus.shape, "nnz", corpus.nnz,
          "seconds", round(time.perf_counter()-started, 1), flush=True)
    del addresses, address_index
    if not query_term_limit:
        del corpus
    output.parent.mkdir(parents=True, exist_ok=True)
    n_queries = n_candidates = 0

    def score_batch(batch, writer):
        nonlocal n_queries, n_candidates
        encoded = vectorizer.transform([address for _, address in batch])
        if sparse_topn:
            from sparse_dot_topn import sp_matmul_topn
            similarities = sp_matmul_topn(encoded, transpose, top_n=100,
                                          n_threads=search_threads, sort=True)
        elif query_term_limit:
            term_rows, term_cols, term_values = [], [], []
            for j in range(encoded.shape[0]):
                lo, hi = encoded.indptr[j:j+2]
                values = encoded.data[lo:hi]
                indices = encoded.indices[lo:hi]
                k_terms = min(query_term_limit, len(values))
                if k_terms:
                    chosen = np.argpartition(values, -k_terms)[-k_terms:]
                    term_rows.extend([j] * k_terms)
                    term_cols.extend(indices[chosen])
                    term_values.extend(values[chosen])
            limited = csr_matrix((np.asarray(term_values, dtype=np.float32),
                                  (term_rows, term_cols)), shape=encoded.shape)
            similarities = (limited @ transpose).tocsr()
        else:
            similarities = (encoded @ transpose).tocsr()
        for j, (query_id, _) in enumerate(batch):
            lo, hi = similarities.indptr[j:j+2]
            values = similarities.data[lo:hi]
            indices = similarities.indices[lo:hi]
            if query_term_limit and len(values) > shortlist_size:
                selected = np.argpartition(values, -shortlist_size)[-shortlist_size:]
                indices = indices[selected]
            if query_term_limit and len(indices):
                values = (corpus[indices] @ encoded[j].T).toarray().ravel()
            # Keep the same 100-address ranking used in the held-out probe.
            # Sparse scores have many exact ties; changing argpartition's k
            # changes tie selection even for the first 20 target IDs.
            k = min(100, len(values))
            ids = []
            if k:
                best = np.argpartition(values, -k)[-k:]
                best = best[np.argsort(-values[best], kind="stable")]
                for address_idx in indices[best]:
                    ids.extend(sorted(ids_by_address[int(address_idx)]))
                    if len(ids) >= top_k:
                        break
            ids = ids[:top_k]
            writer.writerow([query_id, ",".join(ids)])
            n_queries += 1
            n_candidates += len(ids)
        if n_queries % 20_000 < len(batch):
            print("queries", n_queries, "candidate_ids", n_candidates,
                  "seconds", round(time.perf_counter()-started, 1), flush=True)

    query_path = data_dir / f"{data_dir.name}_source1.tsv"
    with query_path.open(encoding="utf-8", newline="") as stream, \
         output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        batch = []
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["country"] != country or (
                    query_ids is not None and row["entity_id"] not in query_ids):
                continue
            batch.append((row["entity_id"], normalize(anyascii(row["business_address"]))))
            if len(batch) == batch_size:
                score_batch(batch, writer)
                batch.clear()
            if max_queries is not None and n_queries + len(batch) >= max_queries:
                break
        if batch:
            score_batch(batch, writer)
    print("COMPLETE", n_queries, n_candidates,
          "seconds", round(time.perf_counter()-started, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--country", default="India")
    parser.add_argument("--query-term-limit", type=int, default=0)
    parser.add_argument("--shortlist-size", type=int, default=500)
    parser.add_argument("--sparse-topn", action="store_true")
    parser.add_argument("--search-threads", type=int, default=8)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.top_k, args.batch_size,
        set(args.query_ids.read_text(encoding="utf-8").splitlines())
        if args.query_ids else None, args.max_queries, args.country,
        args.query_term_limit, args.shortlist_size, args.sparse_topn,
        args.search_threads)
