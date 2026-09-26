# Next actions and decision gates

_Updated 26 September 2026, about 21:20 IST. Read [FINDINGS.md](FINDINGS.md) for evidence, [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for numeric experiments, and [RULES.md](RULES.md) for the submission contract._

## Current stage

Full baseline inference is **complete**: 1,732,544 test Source 1 rows, 353,929,494 candidate pairs. The final baseline `output/matching_results.tsv` passed our full memory-bounded row/order/ID/subset validator and the organizer's match-ID validator. `output/Vulcans_submission.zip` is complete (2,019,135,805 bytes; 18 members); `output/BASELINE_COMPLETE.json` marks it. The baseline ZIP contains the candidate TSV, final matching TSV, code, model, and methodology. The standalone public-leaderboard upload is `output/matching_results.tsv`. No public score has been reported yet. Do not interpret the local development F0.5 as a leaderboard result.

The planned stronger 27-feature LightGBM rescore is **running** through `analysis/finalize_when_ready.py`, using the saved 353.9M candidate pairs and four CPU workers. It will apply top-11 cap and unique target ownership, validate the output, and build `output/Vulcans_number_submission.zip`. It does not redo retrieval or upload anything. Watch `output/finalization.log` and `output/finalization.err.log`; `output/FINALIZATION_COMPLETE.json` marks full completion. Do not launch a duplicate process. The baseline files and ZIP remain available throughout.

The baseline's full unlabeled French raw output had 22,155 of 259,452 queries above 11 links, including 6,411 above 50 and a maximum of 464. The cap reduced French links from 1,895,103 to 825,642. This is a transfer-risk diagnostic, not a measured French score. France has no labels. The user-provided leaderboard screenshot showed top visible scores around 0.988–0.991; our actual public standing remains unknown.

On held-out local data, the stronger model with global threshold 0.75 and India threshold 0.65 scored 0.89272 macro F0.5 on frozen 2,000-query validation and 0.89761 on all 22,133 development queries after cap and ownership. The packaged baseline scored 0.88331 on full development after its cap, ownership, and US nearby-number veto. The variant omits that fixed veto because it lowered the learned model's local score. Local scores are useful for model selection, but the French transfer and private test remain uncertain.

The two-page approach PDF is `output/pdf/Vulcans_approach_summary.pdf`, built by `analysis/build_approach_pdf.py`. It is a separate portal item, not part of the ZIP. The team leader must submit the chosen final ZIP and PDF before **27 September 2026, 23:59 IST**. Confirm the portal's upload-size limit manually if necessary: the baseline archive is 2.02 GB and no verified limit is recorded here.

## Do next

| Priority | Work | Decision gate |
| --- | --- | --- |
| 1 | Upload the verified baseline `output/matching_results.tsv` to the public leaderboard and record team Vulcans' score, rank, time, and submission quota. | This is the only real test feedback available now; keep the baseline ZIP untouched. |
| 2 | Monitor the ongoing stronger-model rescore, then inspect `output/FINALIZATION_COMPLETE.json`, error log, both ZIPs, country comparison, and validator results. Upload `output/number_results.tsv` as the second public candidate if quota/time allow. | Choose final model primarily from actual public feedback, with development results and French tail as supporting evidence. |
| 3 | Prepare the chosen final ZIP plus two-page PDF for portal submission. The ZIP name inside may remain model-specific; preserve both archives. | Verify portal accepts file size, all mandatory fields, and upload completes before deadline. |
| 4 | If the public scores are much below target and time remains, inspect failure evidence and test one bounded change on frozen development/validation before another full rescore. Prioritize retrieval misses and French generic-name collisions; avoid unmeasured broad changes. | Incremental macro F0.5, country slices, candidate cost, full runtime, and realistic time to submit. |

## Decision discipline and handoff

- Candidate recall and oracle ceilings are upper bounds, not model performance.
- No external business lookup, geocoding, data augmentation, or remote inference on supplied records. Respect the package/license constraints in [RULES.md](RULES.md).
- France has no ground truth. Do not claim French F0.5 from raw counts, and keep the baseline archive as a recovery option.
- Keep raw TSVs, extracted records, Parquet, full outputs, local indexes, and credentials out of Git. Commit aggregate findings, scripts, methodology, and small reproducible model weights.
- The repository root is `C:\Users\ASUS\Desktop\CS\Projects\Amazon ML Challenge`. The ignored challenge data are under `6ab10eb3b23ba_student_resource/student_resource/dataset/`. Inspect `git status --short --branch`, this file, and the experiment log on restart.
