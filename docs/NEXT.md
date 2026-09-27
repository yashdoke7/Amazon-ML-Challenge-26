# Next actions and handoff

_Updated 27 September 2026, about 16:50 IST. Evidence and method decisions are in [FINDINGS.md](FINDINGS.md); the submission contract is in [RULES.md](RULES.md)._

## Current state: verified submission files; public score 0.9317

The new US character-address top-10 plus India character-name top-5 package is complete in `output/final/`. The user handles portal uploads. Deadline: **27 September 2026, 23:59 IST**.

| Purpose | Local file | Verification |
| --- | --- | --- |
| Leaderboard TSV | `output/final/matching_results.tsv` | 92,862,288 bytes; SHA-256 `4b7b3fd4d530ca6f054eb66444589bb2a0205ed668d98fb3596f84437d0eb012` |
| Final ZIP | `output/final/Vulcans_submission.zip` | 497,281,183 bytes (<512,000,000); SHA-256 `c38d10d5dc376c8cc83c73c78bed2a08ae5981eb45841f1d1a4ab154276e5061` |
| Approach summary | `output/final/Vulcans_approach_summary.pdf` | 7,206 bytes; SHA-256 `c6c8d06fb49fc72926c20f0f5b6cf933c4b397252872835eed142fee000ec8d1` |

The finalizer accepted **1,732,544** Source 1 queries, **83,724,941** last-stage candidate pairs, and **5,463,955** selected links. The streaming row/order/candidate-subset validator and official match-ID validator passed. The 29-member ZIP passed CRC and its embedded matching and candidate TSV hashes match the validated files. Machine-readable manifest: `output/COMBINED_CHAR_REVISION_COMPLETE.json`. The user reported a **0.9317 public leaderboard score** for this exact matching TSV, up from the preceding 0.928661.

The highest **confirmed public score** is now **0.9317**. Its complete verified TSV, ZIP, PDF, and score manifest are preserved in `output/final_public_0p9317_verified/`, with matching and ZIP hashes checked against the final manifest. The prior 0.928661 package remains in `output/final_public_0p928661_verified/`, and the immediately preceding package is also in `output/final_us_both_verified/`.

The new routes improved complete development macro F0.5 **0.936843→0.939906** and independent frozen validation **0.931181→0.933266** relative to the comparable address/name route. These are local scores, not a public leaderboard prediction. France has no labels in train; the public and private result can differ.

## Do next

1. Keep the verified 0.9317 package as the safe submission. The user has uploaded its TSV for public scoring; the ZIP and PDF must be submitted as the challenge requires before the deadline. Do not upload automatically.
2. The bounded group-aware post-match experiment **failed**: its best development threshold lowered macro F0.5 **0.939906→0.938746**, and frozen validation **0.933266→0.932028**. It is not promoted; keep the 0.9317 package. Details are in FINDINGS and `analysis/probe_anchor_group_rule.py`.
3. Stop score revisions unless a distinct method shows a material independent validation gain and can be fully packaged before the deadline. A label-derived ideal S2/S3 group-propagation oracle reached **0.996882** on development and **0.995486** on frozen validation, but those are hypothetical ceilings, not achieved methods or a justification for unvalidated last-minute changes. The expanded candidate-only oracle **0.987535** also assumes perfect decisions and is not an expected leaderboard score.
4. The `finish-vulcans-submission` heartbeat was set to pause after this verified package was built. Check its status before starting duplicate completion work.

## Rules and reproduction

- No external business-identity lookup, registry, map/geocoding API, external data augmentation, or remote inference on supplied record text. Local method research and locally packaged models are allowed under the problem statement's license and size conditions.
- Every test Source 1 needs exactly one results row and one candidate row, including singletons. Every selected ID must be in that query's last-stage candidate list. The ZIP also includes runnable code, pinned dependencies, README, and filled methodology.
- Supplied data are in `6ab10eb3b23ba_student_resource/student_resource/dataset/` and ignored by Git. Do not commit raw records, output TSVs, model weights, or credentials.
- Source, aggregate findings, and these docs are pushed to `https://github.com/yashdoke7/Amazon-ML-Challenge-26`. The executable package includes its own reproduction README.
