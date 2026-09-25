# Research handoff and next decisions

_Updated 25 September 2026, about 23:30 IST. This is the single starting point if the current conversation or model is interrupted. The private repository is `https://github.com/yashdoke7/Amazon-ML-Challenge-26`._

## What we are trying to achieve

Win the Amazon ML Challenge 2026 business entity resolution task as strongly as possible, with top 50 as the immediate competition goal. Given each deduplicated test Source 1 business, return **all** matching test Source 2/3 IDs, possibly none. The private leaderboard uses **macro F0.5 per Source 1**, so false positives, singletons, and complete multi-record groups matter. The exact output contract, archive layout, validator, and fair-play constraints are in [RULES.md](RULES.md). The supplied PDF, local README, and TSVs are under the ignored `6ab10eb3b23ba_student_resource/student_resource/` folder.

**Current stage:** extensive data exploration and candidate-retrieval experiments are complete. A frozen Source 1 validation split and score function exist. **No matching model has been trained, no held-out model F0.5 exists, and no leaderboard submission has been made.** An interrupted optional training-table export was removed from the committed script; any ignored local `analysis/baseline_pairs.parquet` is unverified and should not be treated as a result. We are continuing analysis, with training scheduled only after the candidate and validation choices are clear.

## Facts established from the supplied data

- Training: 2,206,821 S1, 5,034,616 S2, 5,285,603 S3, and 7,638,365 labeled links. Test: 1,732,544 S1 and 9,969,589 S2/S3 targets. Test contains **259,452 France S1**, with no labeled France examples in train.
- The full label audit found all true target IDs exist, each true pair shares country, and each target has one S1 owner. Country can be used as an **open-set equality block**, never as a US/India-only list.
- 5.58% of training S1 are true singletons. Mean true targets per S1 is 3.46; 11.4% have more than five. A final tiny top-k cap would silently drop valid links.
- In 103,685 sampled positive pairs, 11.2% had weak name similarity, 7.9% weak address similarity, 0.94% both weak, 4.4% blank target address, and 7.1% changed name script. India/S2 name-script changes were 22.7%. These definitions are exploratory string metrics, not semantic truth tests.
- Source 3 has many more alias-marker names (`DBA`, `aka`, etc.) and shorter Indian addresses than Source 2. Exact names and even exact addresses can refer to different labeled entities. Legal-suffix stripping, address equality, or transliteration must never by themselves confirm a match.
- French test text has accents and unseen country behavior. There are no French labels with which to claim a French score or quality parity. The design should be language-agnostic where possible and should be stress-tested on held-out scripts and accent transformations. Do not claim universal language quality without evidence.

## Retrieval experiments already run

The main probe uses 5,000 seeded training S1 records with 17,312 true links, searched against **all 10.32 million training targets**. These are candidate-generation measurements, **not a trained matcher score**. All listed candidate counts are for the 5,000 queries.

| Candidate method | Candidate pairs | Link recall | Non-singleton complete-set recall | Oracle macro F0.5 ceiling |
| --- | ---: | ---: | ---: | ---: |
| Basic normalized exact name or address | 53,441 | 29.5% | Not measured | Not measured |
| Core normalized name or basic address | 183,461 | 47.6% | Not measured | Not measured |
| Low rare-token name/address union | 666,797 | 68.5% | Not measured | Not measured |
| Mid rare-token union | 2,123,362 | 78.9% | Not measured | Not measured |
| High rare-token union | 9,365,116 | 89.3% | 78.4% | 0.9371 |
| High union, separate top-100 name **or** address ranking | 863,082 | 87.5% | 73.7% | 0.9297 |
| Previous row plus compact exact name | 907,433 | 89.3% | 75.8% | 0.9465 |
| Previous top-100 row plus core compact exact name | **1,027,502** | **90.9%** | **78.2%** | **0.9579** |

The **oracle ceiling** assumes a perfect future matcher selects every true link that reached the candidate set and rejects all false links. It is an upper bound on this sample, not actual performance. The last route offers a strong cost/coverage tradeoff: the core-name rescue adds 583 true links and 164,420 pairs over the top-100 set. Even so, its naive full-test extrapolation is about **356 million pairs**, so full-scale runtime and further pruning remain real constraints. The 5,000 queries were exploratory; confirm on the frozen development/validation split before relying on them.

High-token recall was US 91.9%, India 85.4%, cross-script 70.5%, blank target address 61.0%, weak address 56.8%, and both-weak 19.9%. A single joint name/address Jaro-Winkler ranker destroyed the blank-address slice: only 5.0% survived its top 100. Separate name and address quotas were substantially safer. Source-specific slices for the **core rescue** have not yet been measured.

Details and reproducible scripts: [DEEP_EDA.md](../analysis/DEEP_EDA.md), [RETRIEVAL_PROBES.md](../analysis/RETRIEVAL_PROBES.md), [RETRIEVAL_MISS_AUDIT.md](../analysis/RETRIEVAL_MISS_AUDIT.md), and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Raw record extracts and aggregate JSON outputs stay local and ignored by Git.

## Why candidates are missed

Fifty-six original missed groups were manually inspected from a reproducible local sample of 100. The most actionable findings are:

1. Choosing only the rarest query tokens can miss an almost identical target when the rare word is a changed suffix, typo, or inserted fragment. The exact compact/core-name rescue directly addressed part of this.
2. Latin S1 names often match Indian-script target names. Address-only retrieval helps, but shorter target addresses, translated state names, and number changes still defeat it.
3. Blank target addresses require a dedicated name route and quota. Generic names make that route dangerous without a later precision check.
4. Domain-style concatenations, alias/trade names, word-order changes, and character corruptions each create failures that a single exact or full-string similarity measure misses.
5. Shared buildings and almost identical nonmatches mean stronger recall routes also add realistic false matches. Final decisions must be learned or calibrated, with specific hard-negative checks.

## Next research targets, in order

| Priority | Experiment and reason | Required decision evidence |
| --- | --- | --- |
| 1 | **Evaluate the current core-rescue candidate route on the frozen development split** from [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md). The 5k exploratory sample is promising but too small for final selection. | Link recall, complete-set recall, oracle macro F0.5, median/p90/p99/max pairs per S1, runtime, memory, US/India and S2/S3 slices, script-change and blank-address slices. |
| 2 | **Analyze incremental missed true links after the core rescue**, not just misses from older routes. Categorize a seeded sample of original record groups. This prevents optimizing already solved cases. | Counts by script change, blank/shortened address, alias/domain, number corruption, source, and both-weak fields; representative local examples. |
| 3 | **Test narrow rescue channels independently:** accent-folded/compact domain and alias views; address components/number-locality combinations; character n-grams; optional local transliteration. These target distinct remaining gaps. | Incremental true links, incremental candidate pairs, complete-set and oracle-ceiling gains, false-collision tails for each channel. Keep the raw fields alongside every derived view. |
| 4 | **Stress-test generalization.** With no French labels, check France candidate volume and text behavior without claiming accuracy. Hold out Indian scripts or transform Latin accents to see whether a route depends on language-specific shortcuts. | Coverage/cost and qualitative failures by script/country; no unmeasured performance claim. |
| 5 | **Only then train a first matching baseline.** Candidate recall must be high enough first. Use source-aware name/address similarities, aliases, numbers, missingness, and hard negatives; respect the fixed split and exclude held-out-owned targets from training negatives. | Actual held-out macro F0.5, singleton accuracy, source/country/noise slices, calibration/threshold curve, runtime. Clearly separate it from oracle ceilings. |
| 6 | **Scale and package.** Stream full test retrieval/scoring, inspect France and large blocks, generate both required TSVs, run the supplied validator, and prepare the methodology/code archive. | Exact row/ID/candidate-subset checks, reproducible commands, pinned requirements, licensing record, full runtime, portal upload result. |

Route additions should be accepted for **measured incremental benefit**, not because they sound multilingual or sophisticated. In particular, do not spend most of the remaining time on a huge model before confirming that the desired links reach it. At the same time, do not mistake a high oracle ceiling for a score: the precision problem remains open.

## Non-negotiable rules and practical handoff

- No external business-identity lookup, registry, map/geocoding API, external data augmentation, or remote inference on supplied record text. General method research is separate. Any final pretrained model weights must meet the MIT/Apache 2.0 and at-most-8B-parameter condition; verify the actual weights license.
- Submission needs one result row and one final-candidate row for **every** test S1, including singletons. Final matches must be a subset of the listed last-stage candidates. The final ZIP also needs runnable code, pinned dependencies, a README, and the filled methodology template. See [RULES.md](RULES.md) for exact names and validator commands.
- The local data path is `6ab10eb3b23ba_student_resource/student_resource/dataset/`; it is ignored by Git. `analysis/*_results.json`, Parquet, model files, and outputs are ignored too. Commit scripts, aggregate findings, and concise documentation; never commit raw records, credentials, or supplied TSVs.
- The current committed state is on `main`; use `git status` and `git log -1 --oneline` before continuing. The root [README](../README.md) and [HANDOFF.md](HANDOFF.md) point to supporting artifacts. The user will contribute observations as they arise; incorporate them into hypotheses and test them against the frozen split.
