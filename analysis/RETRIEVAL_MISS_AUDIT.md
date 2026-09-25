# Retrieval miss audit

_25 September 2026. Based on the seeded 5,000 train Source 1 query probe. This is a diagnostic sample, not a held-out model evaluation._

The token probe saved 50 randomly selected labeled links missed by the high-volume token union and another 50 recovered by that union but discarded by separate name-or-address top-100 pruning. Original records are in the ignored local `token_retrieval_misses_results.json`; they are not committed. Fifty-six of these groups were read in original text during the initial audit. Slice tags below overlap and are descriptive, not mutually exclusive.

| Failure | Observed pattern | Rescue route to test | Main false-positive risk |
| --- | --- | --- | --- |
| Rarest query tokens do not survive | A name can be nearly identical, yet the chosen rare term is a changed legal suffix, typo, inserted word, or reordered fragment. A target may keep only the common business words. | Add normalized full-name and core-name exact keys independently of rare-token selection. Consider more token choices only where blocks stay small. | Generic core names collide across many businesses. |
| Name appears as a domain/compact string | The target may concatenate words and append a web suffix, sometimes with accent/typo. | Compact and accent-folded domain views; character n-gram lookup. | Similar domain stems and generic words. |
| Cross-script Indian names | Latin query names match Devanagari/Tamil/Telugu/Bengali/Kannada/Malayalam target names. Sometimes the address also changes state script or truncates to locality. | Name-independent address component route; optional **local** transliteration or multilingual representation with verified license. | Same building/locality contains unrelated firms. |
| Address is blank or shortened | Target may contain no address, just a house number plus city/state, or a different city spelling. | Reserved name quota, normalized core-name route, source-specific missingness feature. | Short/generic business names recur. |
| Address number or word corruption | House number changes, zeros are inserted, a street/city is misspelled, or order changes. | Character n-grams and parsed number/street/locality features with soft matching. | Numbers alone and common localities have many collisions. |
| Alias and trade name | A target can be a trading name, `DBA` phrase, or seemingly unrelated name at the same location. | Split alias markers; retain separate original and alias views; mine within-source siblings as a carefully validated rescue channel. | Co-located firms must not be merged automatically. |
| Pre-ranking discards clear links | Jaro-Winkler on full strings undervalues reordered addresses and cross-script names. Some exact names still rank below 100 due to many same-name candidates. | Separate name/address quotas, token overlap, normalized core, address components, and field-aware scoring. | Larger candidate set and runtime. |

## Candidate ceiling measured after miss audit

For the 5,000 sampled S1 queries (17,312 true links), the high-volume union retrieves 89.3% of true links. Among S1 with at least one true link, 78.4% have **every** true target in the candidate set. Candidate counts per query: median 1,657; p90 4,057; p99 6,284; maximum 8,714. Its oracle macro F0.5 ceiling, assuming every available true candidate is selected and no false candidate is selected, is **0.9371**. This is an upper bound on this sample, not an achieved model score.

After separate name-or-address top-100 pruning, link recall is 87.5%, complete-set recall is 73.7%, and the oracle macro F0.5 ceiling is **0.9297**. Candidate counts: median 193, p90 198, p99 199, maximum 200. The 0.0074 ceiling loss is meaningful but smaller than the substantial pair-volume reduction. A trained matcher must still learn to reject nearly all candidates, including singletons; these ceilings say nothing about that precision.

## Next decision gate

Test the rescue routes on a frozen S1 holdout with entity-grouped positives. Favor a route only if it increases complete-set recall or oracle F0.5 ceiling for a reasonable candidate cost, especially in India, cross-script, missing-address, and weak-field slices. Then train a precision-focused matcher and select thresholds on actual macro F0.5. For France and unseen languages, only claim robustness supported by stress tests; there are no French labels to establish equal quality.
