"""Bound reverse TF-IDF retrieval of current development candidate misses.

This is a positive-only diagnostic. Query text is the true target record, but
the index contains every training Source 1 record of the same country. It
measures candidate rank, not deployable F0.5 or final matching quality.
"""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from anyascii import anyascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from features import core, normalize  # noqa: E402

TRAIN = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
DEV = ROOT / "tmp/development_full"
QUOTAS = (1, 5, 10, 20, 50, 100)


def load_lists(path, field):
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["source1_entity_id"]: set(filter(None, row[field].split(",")))
                for row in csv.DictReader(stream, delimiter="\t")}


def missing_targets():
    candidates = load_lists(DEV / "tfidf_top20_candidates.tsv", "candidate_entity_ids")
    truth = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q = row["source1_entity_id"]
            if q in candidates:
                truth[q] = set(filter(None, row["matched_entity_ids"].split(",")))
    assert candidates.keys() == truth.keys()
    return {t: q for q in candidates for t in truth[q] - candidates[q]}


def rank_route(country, field, query_rows, owner, s1_ids, s1_rows):
    started = time.perf_counter()
    name_mode = field == "business_name"
    transform = core if name_mode else normalize
    corpus = [transform(anyascii(row[field])) for row in s1_rows]
    if name_mode:
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 4), min_df=2, max_df=0.8,
            max_features=120_000, sublinear_tf=True, dtype=np.float32)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.8,
            max_features=200_000, sublinear_tf=True, dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(corpus)
    transpose = matrix.T.tocsr()
    del corpus, matrix
    id_to_index = {q: i for i, q in enumerate(s1_ids)}
    counts = Counter()
    by_source = {source: Counter() for source in ("S2", "S3")}
    for first in range(0, len(query_rows), 100):
        batch = query_rows[first:first+100]
        encoded = vectorizer.transform([transform(anyascii(row[field])) for row in batch])
        result = sp_matmul_topn(encoded, transpose, top_n=100,
                                    n_threads=8, sort=True)
        for i, row in enumerate(batch):
            true_col = id_to_index[owner[row["target_id"]]]
            lo, hi = result.indptr[i:i+2]
            positions = np.flatnonzero(result.indices[lo:hi] == true_col)
            if len(positions):
                rank = int(positions[0]) + 1
                for quota in QUOTAS:
                    if rank <= quota:
                        counts[quota] += 1
                        by_source[row["source"]][quota] += 1
    return {"country": country, "field": field, "queries": len(query_rows),
            "source_counts": dict(Counter(r["source"] for r in query_rows)),
            "rank_recall": {str(k): counts[k] for k in QUOTAS},
            "rank_recall_by_source": {s: {str(k): c[k] for k in QUOTAS}
                                      for s, c in by_source.items()},
            "seconds": round(time.perf_counter() - started, 1)}


def main():
    owner = missing_targets()
    print("missing targets", len(owner), flush=True)
    con = duckdb.connect(str(ROOT / ".duckdb/train_index.duckdb"), read_only=True)
    con.register("wanted", pd.DataFrame({"target_id": sorted(owner)}))
    targets = con.execute("""SELECT t.target_id,t.business_name,t.business_address,t.country,t.source
        FROM target t JOIN wanted w USING(target_id)""").df().fillna("")
    con.close()
    assert len(targets) == len(owner)
    by_country = {"US": [], "India": []}
    with (TRAIN / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["country"] in by_country:
                by_country[row["country"]].append(row)
    outputs = []
    for country in ("US", "India"):
        s1_rows = by_country[country]
        s1_ids = [r["entity_id"] for r in s1_rows]
        query_rows = targets.loc[targets.country.eq(country)].to_dict("records")
        for field in ("business_name", "business_address"):
            rows = [r for r in query_rows if r[field]]
            result = rank_route(country, field, rows, owner, s1_ids, s1_rows)
            outputs.append(result)
            print("RESULT", json.dumps(result), flush=True)
    output = ROOT / "analysis/reverse_missing_probe_results.json"
    output.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
