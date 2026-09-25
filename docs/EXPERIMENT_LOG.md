# Experiment log

Record the exact split, data universe, seed, candidate budget, runtime, coverage, macro F0.5, and decision for each experiment. A public leaderboard result is supporting evidence, not a replacement for the held-out set.

| ID | Status | Test and data | Result | Decision |
| --- | --- | --- | --- | --- |
| EDA-01 | Complete | Full supplied train/test scan and full label audit | 7,638,365 valid same-country links; 0 reused target IDs; France only in test | Open-set country blocking is supported; preserve full test coverage |
| EDA-02 | Complete | Seeded 30k S1 random sample, 103,685 positive pairs | Weak-name 11.2%; weak-address 7.9%; both weak 0.94%; script change 7.1%; blank target address 4.4% | Independent retrieval routes and corruption slices required |
| EDA-03 | Complete | Seeded 5k S1 sample joined to all train S2/3 on exact name/address | Exact-key union retrieves 17.6% of true links; exact name has 28,587 labeled nonmatches | Exact equality is a seed, not the solution |
| RET-00 | Complete | Seeded 5k S1 sample, full target universe, normalized exact-key probes | Basic name/address union 29.5% edge recall; crude legal-suffix core name/address 47.6% with 183,461 candidates | Fuzzy retrieval is necessary; core key needs volume limits. [Details](../analysis/RETRIEVAL_PROBES.md) |
| RET-01 | In progress | Frozen held-out S1 validation; independent fuzzy name/address candidate routes | Pending | Choose routes by recall, volume, and subgroup coverage |

## Required metrics for RET-01

- Positive-edge recall overall and by US/India, S2/S3, name script change, missing target address, low name similarity, low address similarity, both weak, domain alias, and generic name.
- Complete-set candidate recall: share of S1 rows whose every true target is included. Report singleton candidate volume separately.
- Candidate count per S1 (median, p90, p99, maximum), estimated full-test pair count, CPU time, and disk/RAM used.
- Inspect at least 20 retrieval misses as original record groups. Record whether name, address, number, script, or alias caused the miss.
