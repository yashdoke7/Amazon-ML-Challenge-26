# Next actions and handoff

_Updated 27 September 2026, about 10:00 IST. Read [FINDINGS.md](FINDINGS.md) for evidence, [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for local experiments, and [RULES.md](RULES.md) for the submission contract._

## Current stage: submission files verified

The stronger Vulcans submission is ready locally. **No portal upload has been made.** The old public baseline remains **0.876, rank 1912**; the new public score is unknown until the team uploads its TSV. The local development and frozen-validation macro F0.5 scores for this pipeline are **0.931832** and **0.927518** after cap and exclusive-owner resolution. These are not public leaderboard estimates, particularly because France has no training labels.

The final production run combined a learned top-40 broad candidate set with exact full-index top-20 India address TF-IDF candidates, scored the added pairs using the 35-feature hard-negative model and blank-address specialist, then capped groups at 11 and resolved exclusive target ownership. The three address segments were validated in Source 1 order for all **809,986 India queries**. The final stream validator accepted **1,732,544 test queries**, **77,471,526 last-stage candidate pairs**, and **5,417,386 selected links**, with all selected IDs within each query's candidate list. The organizer match-ID validator also passed.

## Files for the team

| Purpose | Local path | Verification |
| --- | --- | --- |
| Public leaderboard upload | `output/final/matching_results.tsv` | 1,732,544 rows; 92,265,146 bytes; SHA-256 `7fb79ac1013c535eb5267ac938457d3cd22480196f05643f39938c357409e5b2` |
| Final archive upload | `output/final/Vulcans_submission.zip` | 495,161,412 bytes, below 512,000,000; 27 members; ZIP CRC test passed; embedded matching and candidate TSVs match validated files by SHA-256 |
| Approach summary upload | `output/final/Vulcans_approach_summary.pdf` | Two A4 pages, previously rendered and visually checked |

The archive includes `output/matching_results.tsv`, the exact `output/candidate_pairs.tsv` last-stage union, code, model weights, and methodology. Its candidate member matches `output/generalized_compact_candidate_pairs.tsv` by SHA-256 `ec2c1b720b471d7b0f357b6f01c37983fb71e9f4c909a2004644fa2526720053`. The full completion marker is `output/GENERALIZED_COMPLETE.json`; the build log is `output/generalized_pipeline.log`. Raw data, intermediates, and portal files are ignored by Git and stay on this machine.

## Do next

1. Upload **only** `output/final/matching_results.tsv` for the public leaderboard and record Vulcans' score/rank. Compare it with 0.876/rank 1912. The public score is the first direct signal for transfer to test and France.
2. Submit `output/final/Vulcans_submission.zip` and `output/final/Vulcans_approach_summary.pdf` according to the portal instructions before **27 September 2026, 23:59 IST**. Check that each upload finishes and is accepted. The user said they will perform portal uploads.
3. If leaderboard feedback is substantially worse, investigate the result against the saved baseline and local evidence before changing the complete package. The measured candidate oracle on development is 0.978035, so this pipeline is not guaranteed to reach top 50 against visible leaders near 0.99.

The submission uses supplied records and locally packaged models only. No external business lookup or remote inference was used. The repository is `https://github.com/yashdoke7/Amazon-ML-Challenge-26`; its source and compact evidence are pushed, while data and final artifacts remain local.
