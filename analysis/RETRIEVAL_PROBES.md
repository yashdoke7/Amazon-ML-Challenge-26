# Initial candidate-key probes

_25 September 2026. These are exact-key **retrieval** experiments on 5,000 seeded training Source 1 queries against every training Source 2/3 record. They are not matching-model scores or a complete validation pipeline._

The 5,000 queries have 17,312 labeled positive links. Each route requires equal country labels. Every pair produced by a route is checked against the full training truth for that query. The key transformations operate on both sides. `basic` lowercases and replaces punctuation/spacing runs while retaining Unicode letters and marks; `compact` also removes spaces; `core` additionally removes a first-pass list of Latin legal suffix tokens. The core transformation is intentionally crude and must **not** replace raw names.

| Route | Candidate pairs | True links in candidates | Gold-edge recall | True share among candidates |
| --- | ---: | ---: | ---: | ---: |
| Case-insensitive exact name | 30,495 | 1,908 | 11.0% | 6.3% |
| Basic normalized exact name | 51,842 | 3,863 | 22.3% | 7.5% |
| Compact exact name | 54,198 | 4,057 | 23.4% | 7.5% |
| Core compact exact name | 182,092 | 7,230 | 41.8% | 4.0% |
| Case-insensitive exact address | 1,569 | 1,255 | 7.2% | 80.0% |
| Basic normalized exact address | 1,824 | 1,462 | 8.4% | 80.2% |
| Compact Source 1 name = target domain stem | 4,785 | 176 | 1.0% | 3.7% |
| Core compact Source 1 name = target domain stem | 12,258 | 530 | 3.1% | 4.3% |
| **Basic name OR basic address** | **53,441** | **5,100** | **29.5%** | **9.5%** |
| Core name OR basic address | 183,461 | 8,237 | 47.6% | 4.5% |
| Basic name/address OR core domain | 65,665 | 5,596 | 32.3% | 8.5% |

These numbers expose the next bottleneck: good normalization improves candidate recall substantially, but even core-name plus exact-address retrieval misses **52.4%** of positive links. Fuzzy retrieval is mandatory. Legal-suffix removal recovers 3,137 additional true links over the basic name/address union but brings about 130,000 additional candidate pairs. Its largest sampled query block contains **1,331** targets. It is viable only with scoring, selective block handling, and measured resource limits. The domain route adds 496 true links to the basic union, but many more wrong candidates; it is a rescue channel, not an acceptance rule.

## Rare-token retrieval probe

The same 5,000 queries and all 10.32 million training targets were used. For each query field, tokens of at least four Unicode letters/numbers were ordered by target document frequency. A route chose up to 1, 2, or 3 rarest tokens with document-frequency caps of 500, 1,000, or 3,000. Name and address routes were tested independently and combined by union. This is still a **retrieval ceiling**, not matcher precision or F0.5.

| Route | Candidate pairs | True links retrieved | Gold-edge recall |
| --- | ---: | ---: | ---: |
| Rare name token, cap 500, top 1 | 250,619 | 5,597 | 32.3% |
| Rare address token, cap 500, top 1 | 419,895 | 9,151 | 52.9% |
| Low-volume name/address union | 666,797 | 11,856 | 68.5% |
| Name top 2, cap 1,000 | 546,475 | 6,348 | 36.7% |
| Address top 2, cap 1,000 | 1,582,849 | 11,453 | 66.2% |
| Mid-volume union | 2,123,362 | 13,660 | 78.9% |
| Name top 3, cap 3,000 | 3,295,517 | 8,402 | 48.5% |
| Address top 3, cap 3,000 | 6,082,451 | 13,733 | 79.3% |
| High-volume union | 9,365,116 | 15,457 | 89.3% |

The high-volume union averages 1,873 candidates per query and would imply roughly 3.24 billion pairs for the full 1.73 million test queries if this sample were representative. That is too large for an expensive matcher. Its recall is US 91.9%, India 85.4%, cross-script names 70.5%, missing target address 61.0%, weak address 56.8%, and both weak 19.9%. These slices are the immediate retrieval gaps. Address tokens are substantially more productive than name tokens, especially when names change script.

As a first cheap pruning test, DuckDB Jaro-Winkler similarities were calculated for name and address on the high-volume union. A joint score of `0.65 * max(name, address) + 0.35 * min(name, address)` retained 79.8% of gold links in top 20 and 83.1% in top 100. It nearly eliminated the missing-address slice (5.0% recall at top 100). Independently reserving top 100 by **name OR address** retained 87.5% of gold links in 863,082 pairs, including 58.0% of the missing-address slice. Separate field quotas are the safer current pruning design; Jaro-Winkler is only a feasibility score and must not become an acceptance rule.

The next retrieval probe should inspect actual missed positive groups and add complementary routes for alias/script changes and weak addresses, then measure whether they rescue true links at a workable pair budget. Before a final candidate design, report complete-set recall per S1, candidate-count tails, and a frozen validation split. Reproduce this probe with `python analysis/token_retrieval_probe.py`; the aggregate result JSON and raw-record extracts are local-only and ignored by Git.

The first [miss audit](RETRIEVAL_MISS_AUDIT.md) found that rarest-token selection misses even some near-exact name pairs. The high union has 78.4% complete-set recall on non-singletons and an oracle macro-F0.5 ceiling of 0.9371 on the sampled S1. Separate name/address top-100 pruning has 73.7% complete-set recall and a 0.9297 oracle ceiling. These are candidate-only upper bounds, not achieved scores.

## Exact-name rescue after pruning

Independent exact normalized-name keys were unioned with the name-or-address top-100 set. Compact names remove punctuation and spaces. Core compact names also remove a first-pass list of Latin legal suffixes. The result remains a *candidate* set, not confirmed matches.

| Candidate set | Pairs on 5k S1 | Gold-edge recall | Complete-set recall, non-singletons | Oracle macro F0.5 ceiling |
| --- | ---: | ---: | ---: | ---: |
| Name/address top-100 | 863,082 | 87.54% | 73.68% | 0.9297 |
| Top-100 + compact exact name | 907,433 | 89.26% | 75.79% | 0.9465 |
| Top-100 + core compact exact name | 1,027,502 | 90.91% | 78.18% | 0.9579 |

The core-name rescue adds 583 true links and 164,420 total candidate pairs over the pruned set. It surpasses the original 9.37M-pair high token set's oracle ceiling at about 11% of its candidate count. This is strong evidence for complementary routes, although 1.03M pairs on 5k queries still extrapolates to roughly 356M test pairs and requires careful streaming and further pruning. Source and country slice results for this rescue still need measurement on a frozen validation set.

Reproduce with `python analysis/normalization_benchmark.py`. The raw-pair result JSON is local-only and ignored by Git.
