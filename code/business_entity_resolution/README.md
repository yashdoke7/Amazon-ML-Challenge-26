# Vulcans business entity resolution submission

This directory reproduces the final Vulcans pipeline for Amazon ML Challenge 2026 from the organiser-supplied data. It creates the two required files:

* `output/matching_results.tsv` — final matches.
* `output/candidate_pairs.tsv` — exact last-stage candidate set scored by the final matcher.

The checked-in TSVs in the archive are the exact files used for submission. The pipeline uses only the supplied data, bundled locally trained models, and local Python packages. It does not make network requests or use external business data.

## 1. Setup

Extract the archive next to the organiser's `dataset/` and `utils/` directories, so the working directory has this layout:

```text
student_resource/
  dataset/train/ and dataset/test/
  utils/validate_submission.py
  code/business_entity_resolution/
  output/
```

Python 3.11, 32 GB RAM, and a local SSD were used. CPU is sufficient; this route does not require a GPU. From `student_resource/`:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
New-Item -ItemType Directory -Force output | Out-Null
python code/business_entity_resolution/src/build_index.py --data-dir dataset/test --db test_index.duckdb
```

`build_index.py` creates a local DuckDB index over Sources 2 and 3, including normalised fields and name frequencies. The code sets a 20 GB DuckDB limit. The full process can take many hours and creates temporary local indexes; these are not required in the final archive.

## 2. Final inference route

Run the following commands in order. Intermediate files are intentionally retained because each later route merges into the candidate set from the preceding route.

### Core lexical candidates and learned top-40 block

```powershell
python code/business_entity_resolution/src/infer.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/model.joblib --output-dir output --batch-size 5000 --threshold 0.65
Move-Item output/candidate_pairs.tsv output/broad_candidate_pairs.tsv
python code/business_entity_resolution/src/two_stage_rescore.py --data-dir dataset/test --db test_index.duckdb --broad-candidate output/broad_candidate_pairs.tsv --candidate output/base_candidate_pairs.tsv --matching output/base_raw.tsv --matcher-model code/business_entity_resolution/generalized_model.joblib --blank-specialist-model code/business_entity_resolution/blank_frequency_model.joblib --blank-threshold 0.80 --threshold 0.75 --country-threshold India:0.65 --top-k 40 --workers 4
```

`infer.py` makes a broad same-country lexical pool. `two_stage_rescore.py` uses `model.joblib` only as a ranker and retains the top 40 candidates per Source 1 record. The general matcher and blank-address specialist then score this retained set.

### India word-address retrieval

```powershell
python code/business_entity_resolution/src/address_tfidf_retrieval.py --data-dir dataset/test --output output/india_address_candidates.tsv --top-k 20
python code/business_entity_resolution/src/merge_address_candidates.py --data-dir dataset/test --db test_index.duckdb --base-candidate output/base_candidate_pairs.tsv --base-matching output/base_raw.tsv --extra output/india_address_candidates.tsv --candidate output/candidate_pairs.tsv --matching output/india_raw.tsv --model code/business_entity_resolution/generalized_model.joblib --blank-specialist code/business_entity_resolution/blank_frequency_model.joblib --workers 4
python code/business_entity_resolution/src/cap_predictions.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/generalized_model.joblib --input output/india_raw.tsv --output output/india_capped.tsv --cap 11
python code/business_entity_resolution/src/resolve_exclusivity.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/generalized_model.joblib --input output/india_capped.tsv --output output/india_final.tsv
```

### Gated US word-address retrieval

```powershell
Copy-Item output/candidate_pairs.tsv output/india_candidate_pairs.tsv
Copy-Item output/india_raw.tsv output/india_raw_saved.tsv
python code/business_entity_resolution/src/select_retrieval_queries.py --data-dir dataset/test --matching output/india_final.tsv --output output/us_query_ids.txt --country US --max-matches 3
python code/business_entity_resolution/src/address_tfidf_retrieval.py --data-dir dataset/test --output output/us_address_candidates.tsv --country US --top-k 5 --batch-size 200 --query-ids output/us_query_ids.txt --sparse-topn --search-threads 8
python code/business_entity_resolution/src/merge_address_candidates.py --data-dir dataset/test --db test_index.duckdb --base-candidate output/india_candidate_pairs.tsv --base-matching output/india_raw_saved.tsv --extra output/us_address_candidates.tsv --candidate output/candidate_pairs.tsv --matching output/us_raw.tsv --model code/business_entity_resolution/generalized_model.joblib --blank-specialist code/business_entity_resolution/blank_frequency_model.joblib --country US --general-threshold 0.75 --blank-threshold 0.80 --sparse-extra --workers 4
python code/business_entity_resolution/src/cap_predictions.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/generalized_model.joblib --input output/us_raw.tsv --output output/us_capped.tsv --cap 11
python code/business_entity_resolution/src/resolve_exclusivity.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/generalized_model.joblib --input output/us_capped.tsv --output output/us_final.tsv
```

### Gated US character-name retrieval

```powershell
Copy-Item output/candidate_pairs.tsv output/us_address_candidate_pairs.tsv
Copy-Item output/us_raw.tsv output/us_address_raw_saved.tsv
python code/business_entity_resolution/src/select_retrieval_queries.py --data-dir dataset/test --matching output/india_final.tsv --output output/us_name_query_ids.txt --country US --max-matches 2
python code/business_entity_resolution/src/address_tfidf_retrieval.py --data-dir dataset/test --output output/us_name_candidates.tsv --country US --name-mode --top-k 10 --batch-size 200 --query-ids output/us_name_query_ids.txt --sparse-topn --search-threads 8
python code/business_entity_resolution/src/merge_address_candidates.py --data-dir dataset/test --db test_index.duckdb --base-candidate output/us_address_candidate_pairs.tsv --base-matching output/us_address_raw_saved.tsv --extra output/us_name_candidates.tsv --candidate output/candidate_pairs.tsv --matching output/us_both_raw.tsv --model code/business_entity_resolution/generalized_model.joblib --blank-specialist code/business_entity_resolution/blank_frequency_model.joblib --country US --general-threshold 0.75 --blank-threshold 0.80 --sparse-extra --workers 4
```

### US character-address and India character-name retrieval

```powershell
Copy-Item output/candidate_pairs.tsv output/us_both_candidate_pairs.tsv
Copy-Item output/us_both_raw.tsv output/us_both_raw_saved.tsv
python code/business_entity_resolution/src/char_address_retrieval.py --data-dir dataset/test --split test --country US --field business_address --top-k 10 --term-limit 16 --shortlist 300 --output output/us_char_address_candidates.tsv
python code/business_entity_resolution/src/merge_address_candidates.py --data-dir dataset/test --db test_index.duckdb --base-candidate output/us_both_candidate_pairs.tsv --base-matching output/us_both_raw_saved.tsv --extra output/us_char_address_candidates.tsv --candidate output/candidate_pairs.tsv --matching output/us_char_raw.tsv --model code/business_entity_resolution/generalized_model.joblib --blank-specialist code/business_entity_resolution/blank_frequency_model.joblib --country US --general-threshold 0.75 --blank-threshold 0.80 --sparse-extra --workers 4
Copy-Item output/candidate_pairs.tsv output/us_char_candidate_pairs.tsv
Copy-Item output/us_char_raw.tsv output/us_char_raw_saved.tsv
python code/business_entity_resolution/src/char_address_retrieval.py --data-dir dataset/test --split test --country India --field business_name --top-k 5 --term-limit 16 --shortlist 300 --output output/india_char_name_candidates.tsv
python code/business_entity_resolution/src/merge_address_candidates.py --data-dir dataset/test --db test_index.duckdb --base-candidate output/us_char_candidate_pairs.tsv --base-matching output/us_char_raw_saved.tsv --extra output/india_char_name_candidates.tsv --candidate output/candidate_pairs.tsv --matching output/combined_char_raw.tsv --model code/business_entity_resolution/generalized_model.joblib --blank-specialist code/business_entity_resolution/blank_frequency_model.joblib --country India --general-threshold 0.65 --blank-threshold 0.80 --sparse-extra --workers 4
python code/business_entity_resolution/src/cap_predictions.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/generalized_model.joblib --input output/combined_char_raw.tsv --output output/combined_char_capped.tsv --cap 11
python code/business_entity_resolution/src/resolve_exclusivity.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/generalized_model.joblib --input output/combined_char_capped.tsv --output output/matching_results.tsv
```

The candidate file is the union of the learned top-40 candidates and all accepted TF-IDF additions. The cap limits a predicted Source 1 group to 11 records. Exclusive-owner resolution assigns a target to the highest-scoring Source 1 prediction.

## 3. Validation

The submission-specific streaming validator checks row coverage, IDs, and that every final match belongs to the same row's candidate set without loading the whole candidate TSV into RAM:

```powershell
python code/business_entity_resolution/src/stream_validate.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/__no_candidates__.tsv --test-dir dataset/test --check-ids
```

The organiser's full candidate check can require more than 32 GB RAM at this scale. `stream_validate.py` performs the candidate-subset check in bounded memory; the organiser validator command above checks matching-file IDs and format.

## 4. Rebuilding models

Bundled model files reproduce the submitted route. To rebuild the two final models from the supplied labels, first build the training index, then run:

```powershell
python code/business_entity_resolution/src/build_index.py --data-dir dataset/train --db train_index.duckdb
python code/business_entity_resolution/src/train_hard_negative.py --data-dir dataset/train --db train_index.duckdb --base-model code/business_entity_resolution/generalized_seed_model.joblib --model-out code/business_entity_resolution/generalized_rebuilt.joblib --summary-out output/hard_negative_training_summary.json --sample-size 200000
python code/business_entity_resolution/src/train_blank_frequency.py --data-dir dataset/train --db train_index.duckdb --model-out code/business_entity_resolution/blank_frequency_rebuilt.joblib --summary-out output/blank_frequency_training_summary.json --sample-size 5000
```

The generated weights can differ slightly across platforms; the bundled frozen models are included for exact reproduction of the submitted files.
