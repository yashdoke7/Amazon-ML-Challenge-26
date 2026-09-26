# Findings and decisions

_Updated 26 September 2026. This is the canonical record of what we found, what each experiment measured, and why decisions changed. For the ordered work queue, read [NEXT.md](NEXT.md). The private repository is `https://github.com/yashdoke7/Amazon-ML-Challenge-26`._

## What we are trying to achieve

Win the Amazon ML Challenge 2026 business entity resolution task as strongly as possible, with top 50 as the immediate competition goal. Given each deduplicated test Source 1 business, return **all** matching test Source 2/3 IDs, possibly none. The private leaderboard uses **macro F0.5 per Source 1**, so false positives, singletons, and complete multi-record groups matter. The exact output contract, archive layout, validator, and fair-play constraints are in [RULES.md](RULES.md). The supplied PDF, local README, and TSVs are under the ignored `6ab10eb3b23ba_student_resource/student_resource/` folder.

**Current stage:** extensive data exploration and candidate-retrieval experiments are complete. A frozen Source 1 validation split, score function, and first trained matcher exist. The first model has been evaluated on a 2,000-entity validation sample, but there is no full-validation or leaderboard score yet. The old ignored local `analysis/baseline_pairs.parquet` was a partial candidate extract and must not be used for the current score; the reproducible complete-core extract is `analysis/full_core_pairs.parquet`.

An end-to-end local inference package now exists at `code/business_entity_resolution/`. It builds a DuckDB target index, reproduces the tested candidate routes, scores pairs with the bundled frozen model, and writes both required TSVs. In a 100-query validation equivalence test, the core route produced exactly **19,773/19,773** expected pairs and the combined route exactly **20,809/20,809**; the first 1,000 vectors from each route were bit-identical to the analysis feature function. The bundled model SHA-256 matches the evaluated model. The first 10,000 test S1 were processed by the full run at about 109 seconds per 5,000-query batch and produced 2,047,629 candidates; these are throughput observations, not quality scores. The complete test run and validator remain pending.

An inspection of the first 20,000 test predictions found a serious generic-name collision tail: **251 of 2,981 French queries** had more than 11 predicted matches, including groups above 300 with different street addresses. In the same slice, 0/7,756 US and 4/9,263 India queries exceeded 11. French target volume per S1 is similar to US/India, so this is a model transfer risk, although France has no truth labels to measure it. A probability-ranked cap at 11 was tested on the fixed 2,000-validation-S1 sample: no predictions exceeded 10 there, so its macro F0.5 remained **0.875357** and no labeled validation pair was lost. The full training label audit found maximum true group size 11. A capped-output postprocessor now exists and was checked on a 2,100-test-query dry run: all final matches remained in the listed candidate sets and maximum group size was 11. **This is a bounded mitigation, not evidence of a French score**; use the same cap across countries and preserve the uncapped output for audit.

## Facts established from the supplied data

- Training: 2,206,821 S1, 5,034,616 S2, 5,285,603 S3, and 7,638,365 labeled links. Test: 1,732,544 S1 and 9,969,589 S2/S3 targets. Test contains **259,452 France S1**, with no labeled France examples in train.
- The full label audit found all true target IDs exist, each true pair shares country, and each target has one S1 owner. Country can be used as an **open-set equality block**, never as a US/India-only list.
- 5.58% of training S1 are true singletons. Mean true targets per S1 is 3.46; 11.4% have more than five. The maximum true target count over all 2,206,821 training S1 is **11** (37 entities have 11); a cap below 11 would drop known valid links.
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

High-token recall was US 91.9%, India 85.4%, cross-script 70.5%, blank target address 61.0%, weak address 56.8%, and both-weak 19.9%. A single joint name/address Jaro-Winkler ranker destroyed the blank-address slice: only 5.0% survived its top 100. Separate name and address quotas were substantially safer. The later full development run confirmed the cross-script ranking gap and the benefit of address-token overlap.

Follow-up slice analysis found the core rescue at US 94.0%, India 86.4%, blank target address 76.3%, cross-script **59.8%**, and both-weak **3.4%**. Enlarging the India address quota to top 500 raises cross-script recall to 66.4% but adds **542,000** pairs for only **117** additional true links. This is an inefficient global fix. The next language/address tests should improve the similarity signal or targeted blocking, rather than just increase Jaro-Winkler quotas. These still need confirmation on the frozen development split.

### Development-partition confirmation

The core-rescue route was rerun over all 22,133 deterministic development S1 rows against all 10.32 million training targets (76,472 labeled links; 93 seconds on the local machine). It produced 4,518,733 candidate pairs, 69,730 true links, 91.18% edge recall, 78.97% non-singleton complete-set recall, and a 0.9617 oracle macro-F0.5 ceiling. Candidate counts were median 195, p90 253, p99 724, maximum 1,527. Slice recall was US 94.19%, India 86.72%, cross-script 62.53%, missing target address 75.21%, weak address 66.46%, and both weak 6.29%.

Adding the India address top-500 quota on this same development set increased the set to 6,921,250 pairs, 91.71% edge recall, and a 0.9638 oracle ceiling. It recovered 404 additional true links for 2.40 million additional pairs. The route remains a useful recall reference, but the quota expansion is rejected as the default because its cost is disproportionate. The core rescue is now the **candidate baseline** for any future matcher; remaining work is targeted rescue and precision, not another generic widening pass.

The first focused rescue, a normalized Source 1 name to target domain-stem block, was tested against the full development baseline. It added 19,807 candidate pairs and 93 true links, raising edge recall from 91.18% to 91.31% and the oracle ceiling from 0.9617 to 0.9624. Keep it as an optional low-cost channel; it is not the main multilingual solution.

An address-number rescue was then tested across all development rows. Normalized numeric tokens were frequency-capped to avoid common-number explosions. The tighter setting (`df<=500`, one selected number token) added 814,609 candidates and 204 true links over core rescue, raising edge recall from 91.18% to 91.45% and the oracle ceiling from 0.9617 to 0.9630. The broader setting (`df<=1,000`, two selected tokens) added 2,460,208 candidates and 431 true links, reaching 91.75% and 0.9643. This is a weak cost/recall tradeoff; keep the number route optional and do not make it the main fix.

An audit of the address-number probe found its candidate arithmetic sound. Its script recorded `total_seconds` before the later rescue routes, so that timing field understated complete runtime; this has been fixed. No trained score is implied by those recall figures.

A focused address rerank then changed the tradeoff. On the full 22,133-row development partition, India-only top-50 ranking by normalized address-token containment added **171,364 candidates and 686 true links** over core rescue. Edge recall rose from 91.18% to **92.08%**, non-singleton complete-set recall from 78.97% to **81.24%**, and the oracle macro-F0.5 ceiling from 0.9617 to **0.9653**. Cross-script recall rose from 62.53% to **70.76%**; both-weak recall rose from 6.29% to **16.55%**. Top-100 added 215,362 more candidates but only 56 more true links than top-50, so top-50 is the preferred tested quota. The full focused probe took 297.6 seconds locally, making full-test runtime a scale risk.

Separately, accent-folded core-name equality added **53,138 candidates and 328 true links** over core rescue on full development, reaching 91.61% edge recall and a 0.9644 oracle ceiling. It helps Latin accent variation at low candidate cost. The overlap and accent routes were measured separately against core; their combined incremental benefit has not been measured. Both remain candidate routes that need a precision-focused matcher.

### First matching model

A local LightGBM matcher was trained on the complete core-route candidate extract from the seeded 5,000-S1 sample. The extract contains 991,451 pairs after removing held-out-owned targets from training negatives, including all 15,738 retrievable true links. Features are local Unicode-normalized name/address similarities, token overlap, address-number agreement, lengths, missingness, and source. The installed LightGBM package identifies its license as MIT; no pretrained weights or external identity data were used. A threshold of **0.65** was chosen on an internal training calibration fold plus development rows.

The first larger validation check used a separately sampled **2,000 validation S1** and all 409,990 core candidates for those queries. It scored **0.8744 macro F0.5** at 95.22% pair precision and 79.22% pair recall: 5,462 true links found, 274 false links, and 1,433 missed links. Singleton accuracy was **75.7%** on 103 true singletons. US macro F0.5 was **0.9064** and India **0.8266**. The candidate oracle on the same 2,000 rows is **0.9614**. Of the 1,433 missed links, **588** never entered the candidate set and **845** were available but rejected by the model. This is a local sample score, not a leaderboard or full-validation result, and the model was evaluated on the core candidate route before the new overlap/accent routes are combined.

The earlier local 204-validation-entity test on a partial extract produced an optimistic 0.8749; using the complete core set reduced it to 0.8633. That partial-extract result is superseded. Error inspection found high-confidence false matches among near-identical businesses at changed house/unit numbers and shared addresses, plus low-confidence true aliases, missing-address records, cross-script names, and number corruptions. These patterns support better calibration and targeted features; they do not justify a blanket exact-number reject rule because many labeled matches have corrupted numbers.

The accent-folded core-name and India address-overlap top-50 channels were combined on the same 2,000 validation S1. The union has **430,900 candidates and 6,418 reachable true links**, versus 409,990 and 6,307 for core alone. Its oracle macro-F0.5 ceiling rose from 0.9614 to **0.9678**. Holding the original core-trained model and its 0.65 threshold fixed, actual macro F0.5 rose only from 0.8744 to **0.8754**: 59 extra true links came with 32 extra false links, and singleton accuracy fell from 75.7% to 74.8%.

Retraining the same feature set on 1,040,124 combined-route training-sample pairs selected a 0.70 calibration threshold. On the same 2,000 validation S1 it scored **0.8730**, with 5,404 true links, 259 false links, and 76.7% singleton accuracy. This is below the core-trained model on the combined candidates. The measured gain is in retrieval; current features/calibration do not reliably convert it to macro F0.5. We retained the core-trained model as the comparison baseline and checked threshold stability on a larger development sample.

A subsequent threshold sweep used a separate 2,000-S1 development sample and its full **433,212 combined-route candidates** (6,413 reachable true links). The core-trained model peaked at threshold **0.775**, development macro F0.5 **0.8792** (versus 0.8770 at 0.65), and 82.1% singleton accuracy. The combined-trained model peaked at **0.70**, development macro F0.5 **0.8839**. Applying the development-selected 0.775 to the unchanged 2,000-S1 validation sample gave the core model **0.8700 on core candidates** (5,239 TP/172 FP; 82.5% singleton accuracy) and **0.8728 on combined candidates** (5,284 TP/180 FP; 82.5% singleton accuracy). Both are below the original 0.65 validation scores. The development optimum is therefore unstable at this sample size; a higher threshold trades away too many valid links. Do not select a threshold from the validation sample. For now, retain the original **core-trained model at 0.65 on combined candidates** as a provisional inference baseline, and report the development/validation discrepancy in any methodology write-up.

Details and reproducible scripts: [DEEP_EDA.md](../analysis/DEEP_EDA.md), [RETRIEVAL_PROBES.md](../analysis/RETRIEVAL_PROBES.md), [RETRIEVAL_MISS_AUDIT.md](../analysis/RETRIEVAL_MISS_AUDIT.md), and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Raw record extracts and aggregate JSON outputs stay local and ignored by Git.

## Why candidates are missed

Fifty-six original missed groups were manually inspected from a reproducible local sample of 100. The most actionable findings are:

1. Choosing only the rarest query tokens can miss an almost identical target when the rare word is a changed suffix, typo, or inserted fragment. The exact compact/core-name rescue directly addressed part of this.
2. Latin S1 names often match Indian-script target names. Address-only retrieval helps, but shorter target addresses, translated state names, and number changes still defeat it.
3. Blank target addresses require a dedicated name route and quota. Generic names make that route dangerous without a later precision check.
4. Domain-style concatenations, alias/trade names, word-order changes, and character corruptions each create failures that a single exact or full-string similarity measure misses.
5. Shared buildings and almost identical nonmatches mean stronger recall routes also add realistic false matches. Final decisions must be learned or calibrated, with specific hard-negative checks.

## Non-negotiable rules and practical handoff

- No external business-identity lookup, registry, map/geocoding API, external data augmentation, or remote inference on supplied record text. General method research is separate. Any final pretrained model weights must meet the MIT/Apache 2.0 and at-most-8B-parameter condition; verify the actual weights license.
- Submission needs one result row and one final-candidate row for **every** test S1, including singletons. Final matches must be a subset of the listed last-stage candidates. The final ZIP also needs runnable code, pinned dependencies, a README, and the filled methodology template. See [RULES.md](RULES.md) for exact names and validator commands.
- The local data path is `6ab10eb3b23ba_student_resource/student_resource/dataset/`; it is ignored by Git. `analysis/*_results.json`, Parquet, model files, and outputs are ignored too. Commit scripts, aggregate findings, and concise documentation; never commit raw records, credentials, or supplied TSVs.
- The current committed state is on `main`; use `git status` and `git log -1 --oneline` before continuing. The root [README](../README.md) points to these two canonical documents. The user will contribute observations as they arise; incorporate them into hypotheses and test them against the frozen split.
