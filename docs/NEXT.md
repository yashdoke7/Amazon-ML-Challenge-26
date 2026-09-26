# Next experiments and decision gates

_Updated 26 September 2026. This is the canonical work queue. Read [FINDINGS.md](FINDINGS.md) for the evidence and reasons behind it. Keep this list current when an experiment finishes, fails, or changes priority._

## Current state

The data audit, retrieval probes, miss audit, and deterministic split are complete. The core candidate route reaches 91.18% link recall and a 0.9617 **oracle** macro-F0.5 ceiling on all 22,133 development S1. India-only address-token overlap top-50 and accent-folded core names increase reachable truth. The first LightGBM matcher scored **0.8744 macro F0.5 on 2,000 validation S1** using core candidates, **0.8754** when scoring the combined candidates unchanged, and **0.8730** after retraining on combined candidates. A larger development threshold sweep favored 0.775 for the core-trained model, but that threshold reduced its validation macro F0.5 to 0.8728 on combined candidates. Retain 0.65 as the provisional core-model threshold; India, singleton calibration, and scaling remain the main gaps. The challenge rules and output contract are in [RULES.md](RULES.md).

The final inference package and frozen model are now in `code/business_entity_resolution/`. The full test run is active locally, writing ignored `output/matching_results.tsv` and `output/candidate_pairs.tsv`; its first two 5,000-query batches each took about 109 seconds. Do not start a second full run. Query its current terminal/session or inspect output file row counts before deciding whether to resume. A 100-query regression check exactly reproduced saved core and combined validation candidates.

## Do next, in order

| Priority | Work and why | Decision gate |
| --- | --- | --- |
| 1 | Continue the already-running full test inference. The challenge has less than two days left and 1.73M test queries imply hundreds of millions of candidates. | Monitor progress and disk, then run the supplied validator on both finished TSVs. If interrupted between batches, use `--resume` as documented in the package README. |
| 2 | Diagnose available-but-rejected true links, false links, and singleton merges. Improve features and hard-negative training using training/development entities; concentrate on India, cross-script names, aliases, shared addresses, and corrupted numbers. | Country/source/noise slices; actual macro F0.5, singleton accuracy, precision/recall. Do not introduce a strict number equality rule. |
| 3 | Inspect residual true-link misses after the combined route. The ignored `analysis/core_rescue_dev_audit_results.json` covers older core-route misses; refresh it. Test alias splitting, targeted character n-grams, or local transliteration only where the missed groups support them. | Incremental true links, candidate cost, hard-slice gains, and false-collision tails. Record aggregate or anonymized findings in Git. |
| 4 | Stress-test France and language transfer. France has no labels; inspect candidate volume, accents, and output completeness. Use held-out Indian scripts or controlled transformations for labeled stress tests. | Observable coverage and failures; no unmeasured France or all-language F0.5 claim. |
| 5 | Package the finished outputs and methodology template. | Supplied validator; every S1 row, target IDs, candidate subset, pinned dependencies, license, reproducibility, runtime, portal submission. |

## Decision discipline

- Treat retrieval recall and oracle ceilings as candidate **upper bounds**, never model results. Keep pair count and tail sizes next to every recall gain.
- Accept new routes for measured incremental benefit, especially on hard slices. A larger quota or complex multilingual model is not valuable merely because it sounds robust.
- Use country equality as an open-set block; do not hard-code US/India and exclude France.
- No external business lookup, geocoding, external data augmentation, or remote inference on supplied records. Verify any eventual model weights meet the license and parameter constraints.
- Update [FINDINGS.md](FINDINGS.md) with results and the reason for each decision, this file with the next action, and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) with a compact numeric row. Avoid adding another general handoff document.

## Resume commands and paths

The repository root is `C:\Users\ASUS\Desktop\CS\Projects\Amazon ML Challenge`. Supplied train and test TSVs are under `6ab10eb3b23ba_student_resource/student_resource/dataset/`, ignored by Git. The current reproducible retrieval probes are `analysis/normalization_benchmark.py` and `analysis/token_retrieval_probe.py`; the split/scorer is `code/business_entity_resolution/src/validation.py`. Run `git status --short --branch` and read the latest [experiment log](EXPERIMENT_LOG.md) before editing. Commit and push scripts plus aggregate findings; keep raw TSVs, extracted records, Parquet, outputs, models, and credentials local.
