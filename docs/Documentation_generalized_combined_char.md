# Amazon ML Challenge 2026 — Methodology

**Team:** Vulcans

**Members:** Yash Kailas Doke, Harsh Jitendra Jain, Ayush Tiwari, Vedant Kaulgekar
**Submitted leaderboard result:** public macro F0.5 = **0.9317**

## 1. Overview

The task is business entity resolution: for each Source 1 business, identify its matching records in Sources 2 and 3, including the possibility of no match. We treated this as a precision-focused, many-to-one linkage problem. The final pipeline first retrieves a limited set of plausible target records, then scores those pairs with locally trained classifiers, and finally makes group-level decisions.

The final submission uses only the supplied training and test files. It does not call external APIs, query business registries, geocode addresses, or use remote inference. All text transforms, TF-IDF vocabularies, indexes, and model weights are created locally from the supplied data.

## 2. Data observations and design choices

The supplied training data contained US and India records; the test data also contains France. France is therefore an unseen country rather than a special case encoded in the model. Every retrieval route requires equal country, but country values are handled as open text values instead of an allowlist. The same normalisation and matching logic is consequently applied to France.

Names and addresses have complementary failure modes. We found spelling changes, legal-suffix variation, accents, punctuation changes, abbreviations, scripts that differ between sources, shared addresses, blank target addresses, and corrupted or different street numbers. Exact equality alone was therefore too restrictive, while a single loose similarity score caused false merges for generic business names and shared buildings.

We normalise text by case-folding, retaining Unicode letters and digits, collapsing punctuation and whitespace, removing common legal suffixes for core-name features, and using an accent-folded and local ASCII-transliterated representation alongside the original Unicode text. The original representation is retained: transliteration is a supporting signal, rather than a replacement for the source text.

## 3. Candidate generation and blocking

Candidate generation is deliberately broader than the final decision. For each Source 1 record, we build a same-country pool from several independent local signals:

1. **Rare-token retrieval.** We retrieve targets that share rare name or address tokens. Each query contributes its least frequent usable tokens; target postings have a document-frequency limit of 3,000 and tokens shorter than four characters are excluded from this route.
2. **Field-specific string ranking.** The retrieved pool is ranked independently by name and address Jaro-Winkler similarity. The top 100 candidates from either field are retained.
3. **Exact compact name keys.** We add equality matches for suffix-normalised compact names and accent-folded compact names.
4. **India address-overlap route.** For India, records with at least two normalised address tokens in common can add candidates. This protects retrieval where names differ strongly across scripts.

All broad candidates are scored by a locally trained 23-feature LightGBM ranker. The highest 40 candidates per Source 1 record, with target ID as a deterministic tie break, form the core last-stage candidate set. On the complete held-out development partition, this learned block retained 70,735 of 70,741 reachable positive links from the broad pool while reducing the number of pairs substantially.

We then add bounded field-specific retrieval routes for cases missed by lexical blocking. These routes are fitted at inference time to the supplied test targets and their results are added to the final candidate TSV before final scoring:

* India word unigram/bigram address TF-IDF: up to 20 target IDs per India query.
* US word unigram/bigram address TF-IDF: up to five target IDs, only for US queries with at most three initially selected matches.
* US character 3–4 gram name TF-IDF: up to ten target IDs, only for US queries with at most two initially selected matches.
* US character 3–4 gram address retrieval: up to ten target IDs. Sixteen strong query terms retrieve a 300-address-key shortlist, then the full query reranks that shortlist.
* India character 3–4 gram name retrieval: up to five target IDs using local transliteration and core-name normalisation.

The gates and quotas were chosen using development data and are fixed at test time. They do not depend on test labels. The last-stage `candidate_pairs.tsv` contains the exact union passed to the final matcher: 83,724,941 pairs across 1,732,544 Source 1 test records.

## 4. Pair scoring

The final matcher is a LightGBM classifier trained locally using labels from the supplied training set. It uses a seeded sample of training-owned Source 1 entities, all reachable positive pairs, model-mined hard negatives, and random negatives. Held-out-owned targets are excluded when sampling training negatives.

The general matcher has 35 features. The feature groups include:

* Unicode-normalised, token-sort, and token-set similarity for names and addresses;
* exact and suffix-normalised name indicators;
* accent-folded and transliterated name/address similarity and containment;
* token overlap and containment;
* address-number agreement, disagreement, and leading-number features;
* field lengths, missing address flags, source indicator, and target-name distinctiveness.

A blank-address specialist is used only when the target address is empty. It augments the general pair features with the local document frequency of the target's normalised core name. This prevents generic names from gaining the same confidence as rare names when address evidence is absent.

The final thresholds are 0.65 for India, 0.75 for all other countries, and 0.80 for the blank-address specialist. The thresholds were selected on a separate labelled development sample. France uses the non-India threshold with no France-specific feature or rule.

## 5. Group decisions

The classifier produces pair scores; the output requires entity groups. We apply two deterministic postprocessing steps:

* At most 11 target records are selected for one Source 1 record, ordered by score. Eleven was the largest true group observed in the supplied training labels.
* A target record is assigned to at most one Source 1 owner, resolving collisions by the higher pair score.

These constraints reduce false positives from generic names and shared addresses. They were selected from the observed training-label structure and evaluated on held-out development data.

## 6. Validation and selection

We used deterministic entity-level development partitions, so that target records related to validation entities were not used as training negatives. We measured both candidate recall/oracle ceilings and the final macro F0.5 after the cap and ownership steps. This matters because a high pairwise score can still create an incorrect group.

On a frozen validation set of 1,994 Source 1 queries, the final successive retrieval additions gave the following macro F0.5 results after final postprocessing:

| Final route | Macro F0.5 |
|---|---:|
| India word-address retrieval | 0.927518 |
| + gated US word-address retrieval | 0.929892 |
| + gated US character-name retrieval | 0.931181 |
| + US character-address retrieval | 0.932633 |
| + India character-name retrieval (final route) | 0.933266 |

On the complete 22,133-query development partition, the final route measured 0.939906 macro F0.5. These local values were used to choose the final pipeline; they are not estimates of private-leaderboard performance. The exact submitted `matching_results.tsv` received a public leaderboard score of 0.9317.

The principal uncertainty is unseen French data and residual aliases or script changes that do not enter the candidate pool. We intentionally did not add external records or pretrained embedding models to address those cases, because the final submission must remain auditable, reproducible from the supplied files, and within the 512 MB archive limit.

## 7. Reproducibility and compliance

The archive includes the exact submitted `matching_results.tsv`, its last-stage `candidate_pairs.tsv`, pinned dependencies, frozen locally trained model files, all source code, and end-to-end commands. The code rebuilds the local DuckDB index and TF-IDF representations from the supplied data. The full route uses CPU sparse retrieval and string features; no GPU is required for reproduction.

The implementation makes no network requests and has no external-business-data dependency. `anyascii`, LightGBM, scikit-learn, and sparse-dot-topn are installed Python packages only; their use is local and their versions are pinned in `requirements.txt`.
