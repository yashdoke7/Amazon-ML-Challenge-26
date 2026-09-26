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
Move-Item output/matching_results.tsv output/matching_results_uncapped.tsv
python code/business_entity_resolution/src/cap_predictions.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/model.joblib --input output/matching_results_uncapped.tsv --output output/matching_results_capped.tsv --cap 11
python code/business_entity_resolution/src/resolve_exclusivity.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/model.joblib --input output/matching_results_capped.tsv --output output/matching_results_owned.tsv
python code/business_entity_resolution/src/veto_nearby_number.py --data-dir dataset/test --db test_index.duckdb --input output/matching_results_owned.tsv --output output/matching_results.tsv
python code/business_entity_resolution/src/stream_validate.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/__no_candidates__.tsv --test-dir dataset/test --check-ids
```

Inference writes one row per test Source 1 entity, including empty match/candidate lists. The final matches are always drawn from that row's last-stage candidate set. If inference stops between complete batches, rerun the same inference command with `--resume`; it checks that both output files have matching row IDs before appending. If a process stopped during the write of one batch, repair or discard that partial batch first. Apply the cap, exclusive-owner resolution, and US number-consistency pass only after inference finishes. The supplied validator holds all candidate IDs in memory and is too large for this machine at full scale; the streaming validator checks both files and their subset relation, while the supplied validator checks final match ID existence.

`src/rescore_candidates.py` can apply a different compatible 23- or 27-feature model to the saved `candidate_pairs.tsv` without rebuilding the index or rerunning retrieval. The optional `number_model.joblib` has 27 features and uses threshold 0.75; for that model apply the cap and exclusive-owner passes but omit `veto_nearby_number.py`. Its optional `--workers 4` uses four CPU processes for exact string features. On a 1,000-query baseline slice, four workers took 4.4 seconds versus 7.9 seconds serially, with identical output; a full rescore still takes hours and needs a separate output path. The current pipeline does not use the GPU.

To generate the optional variant after the baseline output is safely preserved, use separate result paths:

```powershell
python code/business_entity_resolution/src/rescore_candidates.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/number_model.joblib --candidate output/candidate_pairs.tsv --output output/number_results_uncapped.tsv --threshold 0.75 --country-threshold India:0.65 --workers 4
python code/business_entity_resolution/src/cap_predictions.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/number_model.joblib --input output/number_results_uncapped.tsv --output output/number_results_capped.tsv --cap 11
python code/business_entity_resolution/src/resolve_exclusivity.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/number_model.joblib --input output/number_results_capped.tsv --output output/number_results.tsv
```

The repository packager uses the same `candidate_pairs.tsv` for both versions. `python analysis/package_submission.py --team-name Vulcans --check-ids` builds the baseline archive; add `--variant number` to build the separate 27-feature archive with its matching methodology and both required model weights. Submit the version with the stronger measured public score, and give the chosen final archive the organizer's required `<team_name>_submission.zip` name.

## Compact learned block for the 512 MB portal limit

The team observed a 512 MB ZIP upload limit. The broad `output/candidate_pairs.tsv` has 353,929,494 test pairs and makes the original ZIP 2.02 GB. To create a truthful smaller last-stage candidate set, run a **separate frozen 23-feature LightGBM model as a ranking/blocking stage** on the broad lexical union. The top 40 pairs per Source 1, ordered by ranker probability then target ID, are written to `output/compact_candidate_pairs.tsv`. Only those pairs pass to the distinct 27-feature final matching model. The broad file is a regenerable intermediate; the compact file is the candidate set audited with the final matcher. This design retained 70,735/70,741 broad-pool true links and all final selected links on the complete 22,133-query development split; it kept all broad-pool true links on the two separately sampled 2,000-query development and validation sets. France remains unlabeled.

Starting from the broad intermediate produced by `src/infer.py` (the baseline postprocessors are unnecessary for this variant), run:

```powershell
python code/business_entity_resolution/src/two_stage_rescore.py --data-dir dataset/test --db test_index.duckdb --broad-candidate output/candidate_pairs.tsv --candidate output/compact_candidate_pairs.tsv --matching output/compact_number_results_uncapped.tsv --top-k 40 --workers 4 --country-threshold India:0.65
python code/business_entity_resolution/src/cap_predictions.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/number_model.joblib --input output/compact_number_results_uncapped.tsv --output output/compact_number_results_capped.tsv --cap 11
python code/business_entity_resolution/src/resolve_exclusivity.py --data-dir dataset/test --db test_index.duckdb --model code/business_entity_resolution/number_model.joblib --input output/compact_number_results_capped.tsv --output output/compact_number_results.tsv
python code/business_entity_resolution/src/stream_validate.py --matching output/compact_number_results.tsv --candidate output/compact_candidate_pairs.tsv --test-dir dataset/test
python analysis/package_submission.py --team-name Vulcans --check-ids --variant compact_number
```

The compact packager maps these two output TSVs to the required archive paths `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Confirm the final ZIP is under the portal's 512 MB limit before uploading. The public leaderboard accepts the standalone `compact_number_results.tsv` under its required filename `matching_results.tsv`; rename a copy for that upload without modifying the verified source file.

## Method and reproducibility

The index stores Source 2/3 records, Unicode letter/digit token postings with target document frequencies, and compact legal-suffix name keys. For each Source 1, three rare name tokens and three rare address tokens (frequency at most 3,000, minimum length four) form an initial pool. A name or address Jaro-Winkler top 100 quota is applied separately. Exact compact core-name and accent-folded core-name matches are then unioned in. India queries get an additional top 50 normalized address-token overlap route when at least two address tokens overlap. All routes use equal country only; any country label is allowed. Each resulting pair receives 23 features and the bundled LightGBM model predicts a link above threshold 0.65. A separate pass caps anomalously large predicted groups at the 11 highest model scores. The largest true group in the entire supplied training set was 11; the cap did not affect any of the 2,000 sampled validation queries. It mainly addresses generic French names, for which test labels are unavailable. A later pass enforces one predicted Source 1 owner per target, selecting the owner with the highest pair score; this follows the supplied training-label structure. Finally, for a US query that has a selected link at its exact leading address number, selected links whose leading number differs by 1–10 are removed. The latter conservative rule was measured on full development and a separate validation sample; it does not alter India or unlabeled France.

To rebuild `model.joblib` from the supplied training data, first index `dataset/train`, then train on the seeded 5,000 Source 1 sample. Training excludes negative pairs whose target belongs to a held-out Source 1. The deterministic entity split and local score are in `src/validation.py`.

```powershell
python code/business_entity_resolution/src/build_index.py --data-dir dataset/train --db train_index.duckdb
python code/business_entity_resolution/src/train.py --data-dir dataset/train --db train_index.duckdb --model-out code/business_entity_resolution/model_rebuilt.joblib
python code/business_entity_resolution/src/train.py --data-dir dataset/train --db train_index.duckdb --model-out code/business_entity_resolution/number_model_rebuilt.joblib --sample-size 20000 --number-features
```

The bundled weights are the frozen tested model. Rebuilt weights can differ slightly with library/hardware ordering, so use the bundled model for exact output reproduction. Threshold 0.65 is provisional from the local development/validation experiments recorded in the repository's `docs/FINDINGS.md`.
