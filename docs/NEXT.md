# Next actions and handoff

_Updated 27 September 2026, about 11:50 IST. Read [FINDINGS.md](FINDINGS.md) for evidence, [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for local experiments, and [RULES.md](RULES.md) for the submission contract._

## Current stage: submission files verified

The stronger Vulcans submission is ready locally. The user reported its public leaderboard result as **0.926444, rank 2260** (the earlier baseline was **0.876, rank 1912** at its earlier submission time). The local development and frozen-validation macro F0.5 scores for this pipeline are **0.931832** and **0.927518** after cap and exclusive-owner resolution. The public score is close to frozen validation, but France still has no labeled local evaluation. The user will handle final portal uploads.

The final production run combined a learned top-40 broad candidate set with exact full-index top-20 India address TF-IDF candidates, scored the added pairs using the 35-feature hard-negative model and blank-address specialist, then capped groups at 11 and resolved exclusive target ownership. The three address segments were validated in Source 1 order for all **809,986 India queries**. The final stream validator accepted **1,732,544 test queries**, **77,471,526 last-stage candidate pairs**, and **5,417,386 selected links**, with all selected IDs within each query's candidate list. The organizer match-ID validator also passed.

## Files for the team

| Purpose | Local path | Verification |
| --- | --- | --- |
| Public leaderboard upload | `output/final/matching_results.tsv` | 1,732,544 rows; 92,265,146 bytes; SHA-256 `7fb79ac1013c535eb5267ac938457d3cd22480196f05643f39938c357409e5b2` |
| Final archive upload | `output/final/Vulcans_submission.zip` | 495,161,412 bytes, below 512,000,000; 27 members; ZIP CRC test passed; embedded matching and candidate TSVs match validated files by SHA-256 |
| Approach summary upload | `output/final/Vulcans_approach_summary.pdf` | Two A4 pages, previously rendered and visually checked |

The archive includes `output/matching_results.tsv`, the exact `output/candidate_pairs.tsv` last-stage union, code, model weights, and methodology. Its candidate member matches `output/generalized_compact_candidate_pairs.tsv` by SHA-256 `ec2c1b720b471d7b0f357b6f01c37983fb71e9f4c909a2004644fa2526720053`. The full completion marker is `output/GENERALIZED_COMPLETE.json`; the build log is `output/generalized_pipeline.log`. Raw data, intermediates, and portal files are ignored by Git and stay on this machine.

## Do next

1. Preserve the verified 0.926444 TSV/ZIP/PDF as fallback. A recheck of the original problem statement confirms the main objective is a complete Source 1 match set, including singletons, under macro F0.5. The current final candidate pool misses 4,391 development true links and rejects another 5,708 reachable links. The candidate oracle is 0.978035. The proposed 10M-record embedding ANN and ensemble have no measured case for a major gain and do not fit the present ZIP/time budget without additional engineering.
2. **Major revision decision:** the predicted sibling context feasibility probe added only +0.0003 on frozen validation, so do not rebuild around it. A 400k-query hard-negative model is training (PID in `tmp/hard_negative_400k_pid.txt`, log `analysis/hard_negative_400k_training.log`); evaluate its fixed-split gain before any full test rerun. A same-data 35-versus-42-feature LightGBM probe is running (PID in `tmp/extended_feature_probe_pid.txt`, log `analysis/extended_feature_probe.log`), testing compact names, partial similarities, and character trigrams without changing production. Evaluate against development-selected thresholds on frozen validation. These are experiments, not changes to the submitted pipeline.
3. **Measured retrieval fallback:** exact US address TF-IDF top5 on the first 7,824/13,309 development US queries raised full development 0.931832→0.934430; on all 1,197 frozen-validation US queries it raised 0.927518→0.929956. Restricting that route to queries with ≤3 current matches retained 0.934337 development and 0.929892 validation. Full US retrieval is still unproven for runtime; an 8-query-term/500-address shortlist was slower than exact on frozen validation (193.0s vs 183.4s), so do not promote that shortcut. A full production run must fit the deadline and 512 MB ZIP cap.
4. Submit `output/final/Vulcans_submission.zip` and `output/final/Vulcans_approach_summary.pdf` according to the portal instructions before **27 September 2026, 23:59 IST**. Check that each upload finishes and is accepted. The user said they will perform portal uploads. The public rank is time dependent and cannot be compared directly with the earlier rank.

The submission uses supplied records and locally packaged models only. No external business lookup or remote inference was used. The repository is `https://github.com/yashdoke7/Amazon-ML-Challenge-26`; its source and compact evidence are pushed, while data and final artifacts remain local.
