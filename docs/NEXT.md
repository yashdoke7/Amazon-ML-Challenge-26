# Next actions and handoff

_Updated 27 September 2026, about 14:10 IST. Evidence is in [FINDINGS.md](FINDINGS.md); the submission contract is in [RULES.md](RULES.md)._

## Current stage: submission ready

The verified gated US address **plus US name** revision is in `output/final/`. The previous public-scored US address package is preserved in `output/final_us_address_verified/`; the earlier India-only package is in `output/final_india_only_0p926444/`. The user handles all portal uploads. Deadline: **27 September 2026, 23:59 IST**.

| Purpose | Local file | Verification |
| --- | --- | --- |
| Leaderboard TSV | `output/final/matching_results.tsv` | 1,732,544 Source 1 rows; SHA-256 `11cc9d7c784da2ce293ecc82cb1aea4933f0c6b75602b8c65141987d12a8d235` |
| Final ZIP | `output/final/Vulcans_submission.zip` | 476,517,448 bytes, below the 512,000,000-byte limit; SHA-256 `d1b39ef007483036da6cf9a95f074c8f65b00898c344401594315b902c458f3f` |
| Approach summary | `output/final/Vulcans_approach_summary.pdf` | Revised two-page A4 PDF |

The streaming row/order/candidate-subset validator, official match-ID validator, ZIP CRC, and archive member hash checks passed. The final candidate TSV has **79,821,601** pairs covering every test Source 1. Machine-readable manifest: `output/US_BOTH_REVISION_COMPLETE.json`. Source and compact findings are pushed to `https://github.com/yashdoke7/Amazon-ML-Challenge-26`; raw data and output artifacts are local and ignored by Git.

The previous US address version scored **0.92817** on the public leaderboard; the India-only version scored **0.926444**. The new US name route improved frozen local validation **0.929892→0.931181** after the address route, but its public score is unknown until upload. France has no labeled local evaluation. Do not claim a 0.98+ score or top-50 rank from these measurements.

## Do next

1. Upload the revised `matching_results.tsv` to check its public score. The user said they will perform portal uploads.
2. Submit the verified ZIP and PDF before the deadline, and confirm the portal accepted both. If the revised TSV unexpectedly underperforms the previous public score, the complete 0.92817 package is in `output/final_us_address_verified/`.
3. If time remains after securing an accepted submission, evaluate reverse address retrieval. A label-blind 5% US target sample gave only **+0.000174** full-development macro F0.5 after forward name retrieval (37 additional true, two false selected links). This is too narrow to justify replacing the verified files without an independent validation, final-size check, and another complete package verification. The reverse name slice gave just +0.000041 and was stopped.

## Rules and reproduction

- No external business identity lookup, registry, map/geocoding API, external data augmentation, or remote inference on supplied record text. Local method research and locally packaged models are allowed under the problem statement's license/size conditions.
- Submission needs one matching-results row and one final-candidate row for every test Source 1, including singletons. Every predicted match must appear in that query's final candidate list. ZIP also includes runnable code, dependencies, README, and filled methodology.
- Data path: `6ab10eb3b23ba_student_resource/student_resource/dataset/`. It is ignored by Git. Do not commit supplied records, output TSVs, trained weights, or credentials.
- The current pipeline uses a learned top-40 lexical blocker, India address top-20, gated US address top-5, gated US name top-10, a general LightGBM matcher, a blank-address specialist, and cap/owner resolution. See [FINDINGS.md](FINDINGS.md) and `docs/Documentation_generalized_us_both.md` for measured tradeoffs and exact method.
