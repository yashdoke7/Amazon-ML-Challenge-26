# Next experiments and decision gates

_Updated 25 September 2026. This is the canonical work queue. Read [FINDINGS.md](FINDINGS.md) for the evidence and reasons behind it. Keep this list current when an experiment finishes, fails, or changes priority._

## Current state

The full data audit, exploratory retrieval experiments, first miss audit, and deterministic entity split are complete. The strongest tested candidate route combines separate name/address top-100 ranking with a core compact-name exact rescue. On 5,000 exploratory S1 queries it reaches 90.9% link recall and a 0.9579 **oracle** macro-F0.5 ceiling in 1.03 million pairs. It remains weak on cross-script and both-weak pairs. No matcher has been trained and no actual held-out F0.5 exists. The challenge rules and output contract are in [RULES.md](RULES.md).

## Do next, in order

| Priority | Work and why | Decision gate |
| --- | --- | --- |
| 1 | Run the core-rescue route on the frozen development split from [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md). The initial 5k sample established feasibility but must be confirmed. | Report link recall, complete-set recall, oracle macro F0.5, candidates/query p50/p90/p99/max, runtime/memory, US/India, S2/S3, cross-script, missing-address, weak-field slices. If scale is prohibitive, run a deterministic subset of the development split and say so. |
| 2 | Inspect **remaining** true-link misses after core rescue, rather than older-route misses. The latest 60 sampled misses are saved locally in ignored `analysis/core_rescue_audit_results.json`. | Categorize original records by script, blank/shortened address, alias/domain, number corruption, and source. Check at least 20; record only aggregate or anonymized findings in Git. |
| 3 | Test focused rescue channels one at a time: accent-folded/compact domain names, alias splitting, address components and number/locality combinations, character n-grams, and possibly local transliteration. The existing India address-top500 expansion was too costly. | Measure **incremental** true links, pairs, complete-set recall, oracle ceiling, and false-collision tails versus the core-rescue set. Retain raw fields alongside normalized views. |
| 4 | Stress-test France and language transfer. France has no labels; evaluate candidate volume, accents, script behavior, and output completeness. Use held-out Indian scripts or controlled transformations for labeled stress tests. | Record observable coverage and failure patterns. Do not claim an unmeasured France or all-language F0.5. |
| 5 | Train a precision-focused baseline after candidate design is sufficiently strong. Use hard negatives, source-aware features, aliases, address components, missingness, and calibrated thresholds. | Actual held-out macro F0.5, singleton accuracy, precision/recall, country/source/noise slices, runtime. Exclude held-out-owned targets from training negatives. |
| 6 | Scale and package full test inference. Stream candidate generation and scoring, then generate both required TSVs and final archive. | Run supplied validator; check every S1 row, target IDs, candidate subset, pinned dependencies, license, reproducibility, runtime, and portal submission. |

## Decision discipline

- Treat retrieval recall and oracle ceilings as candidate **upper bounds**, never model results. Keep pair count and tail sizes next to every recall gain.
- Accept new routes for measured incremental benefit, especially on hard slices. A larger quota or complex multilingual model is not valuable merely because it sounds robust.
- Use country equality as an open-set block; do not hard-code US/India and exclude France.
- No external business lookup, geocoding, external data augmentation, or remote inference on supplied records. Verify any eventual model weights meet the license and parameter constraints.
- Update [FINDINGS.md](FINDINGS.md) with results and the reason for each decision, this file with the next action, and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) with a compact numeric row. Avoid adding another general handoff document.

## Resume commands and paths

The repository root is `C:\Users\ASUS\Desktop\CS\Projects\Amazon ML Challenge`. Supplied train and test TSVs are under `6ab10eb3b23ba_student_resource/student_resource/dataset/`, ignored by Git. The current reproducible retrieval probes are `analysis/normalization_benchmark.py` and `analysis/token_retrieval_probe.py`; the split/scorer is `code/business_entity_resolution/src/validation.py`. Run `git status --short --branch` and read the latest [experiment log](EXPERIMENT_LOG.md) before editing. Commit and push scripts plus aggregate findings; keep raw TSVs, extracted records, Parquet, outputs, models, and credentials local.
