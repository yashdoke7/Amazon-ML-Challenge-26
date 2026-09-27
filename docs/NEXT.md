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

The previous US address version scored **0.92817** on the public leaderboard; the India-only version scored **0.926444**. The user reported **0.928661** for the current US address plus name route, a gain of just **0.000491** over the US-address version. Frozen local validation moved **0.929892→0.931181**, so the new route transferred in direction but less in magnitude. France has no labeled local evaluation. Do not claim a 0.98+ score or top-50 rank from these measurements.

## Do next

1. Keep the verified `output/final/` package intact while working on a genuinely new candidate and group-decision method. The user will upload the final ZIP and PDF before the deadline and confirm acceptance.
2. Fresh raw-data audit ruled out exact cross-split name+address memorization and row/ID-order shortcuts. An independent error budget on the 22,133-query development split found 4,391 true links absent from current candidates, 5,708 reachable true links rejected by the matcher, and 1,607 false links. Its candidate oracle is **0.978035**. A 0.99 method must change both retrieval and matching; minor quota or threshold changes cannot bridge the gap.
3. Fresh US character-address top ten and India character-name top five together raised complete development macro F0.5 **0.936843→0.939906** and independent frozen validation **0.931181→0.933266** after the current address/name routes. The measured fast search uses 16 strong character terms, a 300-key shortlist, and full-vector reranking. Full US test retrieval is running in session **48657**, output `output/us_char_address_top10.tsv`, and must print `COMPLETE 663106`; full India test retrieval is session **56409**, output `output/india_char_name_top5.tsv`, and must print `COMPLETE 809986`. Both outputs are partial until then. The current verified package stays in `output/final/`.
4. Once both searches finish, run `python analysis/run_combined_char_revision.py` from the repository root. It verifies input row counts/order, merges/scorers US then India onto the current uncapped raw result, caps/resolves ownership, validates both TSVs and match IDs, packages the code/methodology, checks ZIP CRC and <512,000,000 bytes, backs up the current verified package to `output/final_us_both_verified/`, and only then promotes new TSV/ZIP/PDF to `output/final/`. The prepared methodology is `docs/Documentation_generalized_combined_char.md`, package code is `code/business_entity_resolution/src/char_address_retrieval.py`, and the updated two-page PDF is `output/pdf/Vulcans_approach_summary_combined_char.pdf` (rendered and visually inspected). If India fails or the combined ZIP exceeds size, use the US-only fallback finalizer `analysis/run_us_char_revision.py` with its separate methodology/PDF. Do not upload automatically.
5. Reverse-address retrieval remains an optional comparison: a label-blind 5% US target sample gave only **+0.000174** full-development macro F0.5 after forward name retrieval (37 true, two false added). It is too narrow to justify replacing the verified package without independent validation and final-size check. Reverse-name gave +0.000041.

## Rules and reproduction

- No external business identity lookup, registry, map/geocoding API, external data augmentation, or remote inference on supplied record text. Local method research and locally packaged models are allowed under the problem statement's license/size conditions.
- Submission needs one matching-results row and one final-candidate row for every test Source 1, including singletons. Every predicted match must appear in that query's final candidate list. ZIP also includes runnable code, dependencies, README, and filled methodology.
- Data path: `6ab10eb3b23ba_student_resource/student_resource/dataset/`. It is ignored by Git. Do not commit supplied records, output TSVs, trained weights, or credentials.
- The current pipeline uses a learned top-40 lexical blocker, India address top-20, gated US address top-5, gated US name top-10, a general LightGBM matcher, a blank-address specialist, and cap/owner resolution. See [FINDINGS.md](FINDINGS.md) and `docs/Documentation_generalized_us_both.md` for measured tradeoffs and exact method.
