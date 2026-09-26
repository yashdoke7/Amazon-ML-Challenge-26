# Amazon ML Challenge 2026 entity resolution

This package generates both required test TSVs using the supplied data and a bundled LightGBM matcher. The model in `model.joblib` was trained locally on the supplied training TSVs. It uses 23 local string/number/source features; it makes no network calls. LightGBM 4.5.0 is MIT licensed. See `src/train.py` to regenerate model weights from the supplied training set.

## Inputs and environment

Expected layout, relative to `student_resource/`:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Python 3.11 was used locally. Install `requirements.txt` into a clean environment. The tested machine has 32 GB RAM and a local SSD; `src/build_index.py` sets a 20 GB DuckDB limit and the full test outputs can take several GB. The test index can be rebuilt after interruption; its stages are marked in `index_meta`.

From `student_resource/`, with `code/business_entity_resolution` copied into place:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/build_index.py --data-dir dataset/test --db test_index.duckdb
python code/business_entity_resolution/src/infer.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/model.joblib --output-dir output --batch-size 5000 --threshold 0.65
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Inference writes one row per test Source 1 entity, including empty match/candidate lists. The final matches are always drawn from that row's last-stage candidate set. If inference stops between complete batches, rerun the same command with `--resume`; it checks that both output files have matching row IDs before appending. If a process stopped during the write of one batch, repair or discard that partial batch first.

## Method and reproducibility

The index stores Source 2/3 records, Unicode letter/digit token postings with target document frequencies, and compact legal-suffix name keys. For each Source 1, three rare name tokens and three rare address tokens (frequency at most 3,000, minimum length four) form an initial pool. A name or address Jaro-Winkler top 100 quota is applied separately. Exact compact core-name and accent-folded core-name matches are then unioned in. India queries get an additional top 50 normalized address-token overlap route when at least two address tokens overlap. All routes use equal country only; any country label is allowed. Each resulting pair receives 23 features and the bundled LightGBM model predicts a link above threshold 0.65.

To rebuild `model.joblib` from the supplied training data, first index `dataset/train`, then train on the seeded 5,000 Source 1 sample. Training excludes negative pairs whose target belongs to a held-out Source 1. The deterministic entity split and local score are in `src/validation.py`.

```powershell
python code/business_entity_resolution/src/build_index.py --data-dir dataset/train --db train_index.duckdb
python code/business_entity_resolution/src/train.py --data-dir dataset/train --db train_index.duckdb --model-out code/business_entity_resolution/model_rebuilt.joblib
```

The bundled weights are the frozen tested model. Rebuilt weights can differ slightly with library/hardware ordering, so use the bundled model for exact output reproduction. Threshold 0.65 is provisional from the local development/validation experiments recorded in the repository's `docs/FINDINGS.md`.
