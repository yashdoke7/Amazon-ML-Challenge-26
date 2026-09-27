# Next actions and handoff

_Updated 27 September 2026, about 16:38 IST. Evidence and method decisions are in [FINDINGS.md](FINDINGS.md); the submission contract is in [RULES.md](RULES.md)._

## Current state: verified submission files; new public score pending

The new US character-address top-10 plus India character-name top-5 package is complete in `output/final/`. The user handles portal uploads. Deadline: **27 September 2026, 23:59 IST**.

| Purpose | Local file | Verification |
| --- | --- | --- |
| Leaderboard TSV | `output/final/matching_results.tsv` | 92,862,288 bytes; SHA-256 `4b7b3fd4d530ca6f054eb66444589bb2a0205ed668d98fb3596f84437d0eb012` |
| Final ZIP | `output/final/Vulcans_submission.zip` | 497,281,183 bytes (<512,000,000); SHA-256 `c38d10d5dc376c8cc83c73c78bed2a08ae5981eb45841f1d1a4ab154276e5061` |
| Approach summary | `output/final/Vulcans_approach_summary.pdf` | 7,206 bytes; SHA-256 `c6c8d06fb49fc72926c20f0f5b6cf933c4b397252872835eed142fee000ec8d1` |

The finalizer accepted **1,732,544** Source 1 queries, **83,724,941** last-stage candidate pairs, and **5,463,955** selected links. The streaming row/order/candidate-subset validator and official match-ID validator passed. The 29-member ZIP passed CRC and its embedded matching and candidate TSV hashes match the validated files. Machine-readable manifest: `output/COMBINED_CHAR_REVISION_COMPLETE.json`. No portal score has yet been reported for this revision.

The highest **confirmed public score** remains **0.928661**, preserved separately with its TSV, ZIP, PDF and score manifest in `output/final_public_0p928661_verified/`. The immediately preceding package is also in `output/final_us_both_verified/`. Neither backup was changed by the new finalizer.

The new routes improved complete development macro F0.5 **0.936843→0.939906** and independent frozen validation **0.931181→0.933266** relative to the comparable address/name route. These are local scores, not a public leaderboard prediction. France has no labels in train; the public and private result can differ.

## Do next

1. The user uploads `output/final/matching_results.tsv` to the portal for public feedback and submits the ZIP and PDF as the challenge requires before the deadline. Do not upload automatically.
2. Record the returned score. If it exceeds 0.928661, preserve this complete verified package in a new score-specific backup before changing `output/final/`. If it does not, retain the existing score-specific backup as the known best. A ZIP/hash manifest alone is not a leaderboard score.
3. If there is time for further research, prioritize S2/S3 target-to-target grouping and group-aware decisions. A label-derived ideal group-propagation oracle reached **0.996882** on development and **0.995486** on frozen validation because most groups already have at least one true candidate. These are hypothetical ceilings, not achieved methods; no unverified grouping rule belongs in the submitted file.
4. The `finish-vulcans-submission` heartbeat was set to pause after this verified package was built. Check its status before starting duplicate completion work.

## Rules and reproduction

- No external business-identity lookup, registry, map/geocoding API, external data augmentation, or remote inference on supplied record text. Local method research and locally packaged models are allowed under the problem statement's license and size conditions.
- Every test Source 1 needs exactly one results row and one candidate row, including singletons. Every selected ID must be in that query's last-stage candidate list. The ZIP also includes runnable code, pinned dependencies, README, and filled methodology.
- Supplied data are in `6ab10eb3b23ba_student_resource/student_resource/dataset/` and ignored by Git. Do not commit raw records, output TSVs, model weights, or credentials.
- Source, aggregate findings, and these docs are pushed to `https://github.com/yashdoke7/Amazon-ML-Challenge-26`. The executable package includes its own reproduction README.
