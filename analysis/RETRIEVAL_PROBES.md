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

The next probe is token/character-based retrieval with a candidate-volume budget, followed by full held-out recall by risk slice. Retrieval must work across the noisy name and address views; no exact-key rule can be the final blocker.

Reproduce with `python analysis/normalization_benchmark.py`. The raw-pair result JSON is local-only and ignored by Git.
