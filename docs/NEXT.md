# Next actions and handoff

_Updated 27 September 2026, about 13:12 IST. Read [FINDINGS.md](FINDINGS.md) for evidence and [RULES.md](RULES.md) for the submission contract._

## Current stage: submission files verified

The **gated US-address revision** is now verified and ready locally. Its frozen-validation macro F0.5 is **0.929892** versus 0.927518 for the India-only predecessor. The latter received public **0.926444, rank 2260**; the new revision has no public score yet. The user will handle portal uploads. France has no labeled local evaluation.

The verified archive includes the learned top-40 block, India address top-20, and a same-country US address top-5 route for **399,606** US queries with at most three initial matches. The last-stage candidate TSV has **78,408,708** pairs for all **1,732,544** Source 1 queries. The streaming row/order/subset validator, official match-ID validator, and ZIP CRC check passed. The old public-scored version is preserved at `output/final_india_only_0p926444/`.

## Files for the team

| Purpose | Local path | Verification |
| --- | --- | --- |
| Public leaderboard upload | `output/final/matching_results.tsv` | 1,732,544 rows; SHA-256 `7a91723a4293363a9c74f7cf7b45fe091f64b0309c4e9c4f7fe380e5a34261fd` |
| Final archive upload | `output/final/Vulcans_submission.zip` | 500,490,082 bytes, below 512,000,000; 28 members; CRC and both validators passed |
| Approach summary upload | `output/final/Vulcans_approach_summary.pdf` | Revised two-page A4 PDF, rendered and visually checked |

The archive includes `output/matching_results.tsv`, exact `output/candidate_pairs.tsv`, code, model weights, and filled methodology. Its candidate member matches `output/generalized_us_address_candidate_pairs.tsv` by SHA-256 `e2e376a06d03412ac78d9b67325a5b3db89575c7f2b8d44541bc1d4e07b84e02`. The new completion marker is `output/US_ADDRESS_REVISION_COMPLETE.json`. Raw data and portal files stay local and are ignored by Git.

## Do next

1. Preserve the verified US-address TSV/ZIP/PDF in `output/final/` while the final name experiment runs. The India-only public-scored fallback is in `output/final_india_only_0p926444/`. The competition objective remains macro F0.5 over complete Source 1 groups, including singletons; the original India-only candidate oracle was 0.978035 on development.
2. **Major revision decision:** the predicted sibling context probe added only +0.0003 on frozen validation. Replacing the 23-feature top40 blocker with the final matcher or a hybrid did not improve the older 2k/2k samples; pruning did not lose their reachable true links. The 42-feature model gained +0.00788 versus a same-data 35-feature small model, but a 25% blend with the current larger matcher gained only +0.00188 on the older frozen 2k validation pool. The 400k-query retrain finished (10.49m kept pairs), but scored 0.918969 vs 0.920650 packaged reference on development and 0.910809 vs 0.910586 on fixed validation. **Reject full production rescore of this model.**
3. **Anchor rescue measured but deferred:** On all 2,203 development queries with exactly one existing match, one-hop predicted-anchor top-5 + a development-selected dual-probability rule raised subgroup macro F0.5 0.834692→0.841476, worth only about +0.0007 when weighted across all queries before cap/owner. The all-query route costs many more anchors. This is not the major revision to run now. Data and code are in `docs/FINDINGS.md`, `analysis/anchor_retrieval_probe.py`, and `analysis/evaluate_anchor_rescue.py` if the faster retrieval options disappoint.
4. **Active name route:** a full-index US character 3-4-gram TF-IDF search (PID 36572; unified shell session 95564) is writing `output/us_name_tfidf_top10_gated.tsv` for **237,217** US test queries with at most two matches after the India stage. The exact gate is `tmp/us_name_gate2_ids.txt`. **Do not treat the output as complete until the process exits with `COMPLETE 237217`.** The development-selected name top-10/≤2 rule alone raised full-development 0.931832→0.934029. On the frozen 1,994-query sample, it added seven true and zero false links after the US-address route, raising **0.929892→0.931181**. The archived name score is still unknown. `docs/Documentation_generalized_us_both.md` has a candidate-count placeholder until final packaging. The revised two-page `output/pdf/Vulcans_approach_summary_both.pdf` has been rendered and visually checked.
5. **Finish if the name search completes:** run `python analysis/run_us_name_revision.py` from repository root. It validates all 237,217 name rows, merges onto the completed US-address **candidate set and uncapped raw results**, caps/owns, runs both validators, packages with compression level 9, and asserts <512,000,000 bytes before updating `output/final/`. It backs up the current verified US-address package at `output/final_us_address_verified/`. If the ZIP exceeds the limit, leave `output/final/` untouched and test a smaller name quota or gate using the complete retrieved top-10 TSV. This script is prepared but has not yet been run.
6. Submit `output/final/Vulcans_submission.zip` and `output/final/Vulcans_approach_summary.pdf` according to the portal instructions before **27 September 2026, 23:59 IST**. Check that each upload finishes and is accepted. The user said they will perform portal uploads. The public rank is time dependent and cannot be compared directly with the earlier rank.

The submission uses supplied records and locally packaged models only. No external business lookup or remote inference was used. The repository is `https://github.com/yashdoke7/Amazon-ML-Challenge-26`; its source and compact evidence are pushed, while data and final artifacts remain local.
