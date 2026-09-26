# Next experiments and decision gates

_Updated 26 September 2026. This is the canonical work queue. Read [FINDINGS.md](FINDINGS.md) for the evidence and reasons behind it. Keep this list current when an experiment finishes, fails, or changes priority._

## Current state

The data audit, retrieval probes, miss audit, and deterministic split are complete. The core candidate route reaches 91.18% link recall and a 0.9617 **oracle** macro-F0.5 ceiling on all 22,133 development S1. India-only address-token overlap top-50 improves this to 92.08% recall and a 0.9653 oracle ceiling with 171,364 extra candidates. Accent-folded core names separately add 328 true links. The first LightGBM matcher scored **0.8744 macro F0.5 on 2,000 validation S1** using the core route; India and singleton performance are the main gaps. The challenge rules and output contract are in [RULES.md](RULES.md).

## Do next, in order

| Priority | Work and why | Decision gate |
| --- | --- | --- |
| 1 | Combine the tested India address-overlap top-50 and accent-folded core-name channels, measure their *joint* candidate gain, and score those extra pairs with the saved matcher. The two gains were measured separately and the current model score uses core candidates only. | Joint link and complete-set recall, oracle ceiling, pair count/tails, actual macro F0.5 on the 2,000-row validation sample, and runtime. Retain the exact candidate set passed to the model. |
| 2 | Diagnose the 845 available-but-rejected links, 274 false links, and 25 singleton false merges in the large validation sample. Improve local features/calibration against a development sample, then evaluate validation sparingly. | Country/source/noise slices; macro F0.5, singleton accuracy, and precision/recall. Check near-identical wrong entities at shared addresses and true links with corrupted numbers or blank fields. |
| 3 | Inspect residual true-link misses after the combined route. The ignored `analysis/core_rescue_dev_audit_results.json` covers older core-route misses; refresh it. Test alias splitting, targeted character n-grams, or local transliteration only where the missed groups support them. | Incremental true links, candidate cost, hard-slice gains, and false-collision tails. Record aggregate or anonymized findings in Git. |
| 4 | Stress-test France and language transfer. France has no labels; inspect candidate volume, accents, and output completeness. Use held-out Indian scripts or controlled transformations for labeled stress tests. | Observable coverage and failures; no unmeasured France or all-language F0.5 claim. |
| 5 | Scale and package full test inference. Stream retrieval and scoring, produce both required TSVs and the final archive. | Supplied validator; every S1 row, target IDs, candidate subset, pinned dependencies, license, reproducibility, runtime, portal submission. |

## Decision discipline

- Treat retrieval recall and oracle ceilings as candidate **upper bounds**, never model results. Keep pair count and tail sizes next to every recall gain.
- Accept new routes for measured incremental benefit, especially on hard slices. A larger quota or complex multilingual model is not valuable merely because it sounds robust.
- Use country equality as an open-set block; do not hard-code US/India and exclude France.
- No external business lookup, geocoding, external data augmentation, or remote inference on supplied records. Verify any eventual model weights meet the license and parameter constraints.
- Update [FINDINGS.md](FINDINGS.md) with results and the reason for each decision, this file with the next action, and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) with a compact numeric row. Avoid adding another general handoff document.

## Resume commands and paths

The repository root is `C:\Users\ASUS\Desktop\CS\Projects\Amazon ML Challenge`. Supplied train and test TSVs are under `6ab10eb3b23ba_student_resource/student_resource/dataset/`, ignored by Git. The current reproducible retrieval probes are `analysis/normalization_benchmark.py` and `analysis/token_retrieval_probe.py`; the split/scorer is `code/business_entity_resolution/src/validation.py`. Run `git status --short --branch` and read the latest [experiment log](EXPERIMENT_LOG.md) before editing. Commit and push scripts plus aggregate findings; keep raw TSVs, extracted records, Parquet, outputs, models, and credentials local.
