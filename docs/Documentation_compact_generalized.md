# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Vulcans
**Team Members:** Yash Kailas Doke, Harsh Jitendra Jain, Ayush Tiwari, Vedant Kaulgekar
**Submission Date:** 27 September 2026

## 1. Executive Summary

We resolve each deduplicated Source 1 business to zero or more Source 2/3 records using country-aware lexical retrieval, a learned candidate ranker, and a distinct LightGBM final matcher. Separate name and address routes protect records with missing fields, aliases, Indian script changes, and French accents. The final model learns first-address-number agreement, then applies group-size and single-owner decisions to contain collisions.

## 2. Methodology

### 2.1 Problem Analysis

Training contains 2,206,821 Source 1 queries, 10,320,219 Source 2/3 targets, and 7,638,365 positive links. Test adds France, which has no labeled training examples. In a labeled pair sample, 11.2% had weak name similarity, 7.9% weak address similarity, 4.4% blank target address, and 7.1% changed name script. Identical names and addresses can also belong to distinct labeled entities. A whole-string equality rule or a single name/address similarity score therefore loses true links and creates false merges. The metric is macro F0.5 per Source 1, including empty true groups; we optimized for complete group quality and precision, not pair accuracy.

### 2.2 Solution Strategy

**Approach Type:** Broad lexical retrieval, then a learned top-40 blocking stage, a Unicode-aware final pair classifier, and bounded group postprocessing.
**Core contribution:** Separate rare-token retrieval and field-specific top-100 ranking, combined with compact/accent-folded name equality and targeted Indian address-token overlap. A preliminary 23-feature model ranks the broad union and retains at most 40 per Source 1. All routes use equal country as an open-set block, including France.

## 3. Candidate Generation

We tokenize Unicode letters/digits in names and addresses and choose each query's three least frequent target tokens with document frequency at most 3,000 and length at least four. The union of their postings is ranked independently by name and address Jaro-Winkler similarity; a pair survives if either rank is at most 100. We add compact names after common legal-suffix removal, accent-folded compact names, and, for Indian queries, the top 50 candidates by address-token overlap when at least two tokens overlap. Country must match. This broad union is an intermediate and can be regenerated from the supplied records. A frozen 23-feature LightGBM model scores the broad pairs solely to rank candidates; the highest 40 per Source 1 (ties by target ID) form the **last-stage candidate set** written to `candidate_pairs.tsv` and passed to the separate final matcher. The broad intermediate is not passed directly to that final matcher.

On all 22,133 held-out development Source 1 records, the core route retrieved 91.18% of positive links with 4,518,733 candidate pairs and an oracle macro-F0.5 ceiling of 0.9617. Independently, India address-overlap top 50 added 686 true links for 171,364 more pairs and raised cross-script recall from 62.53% to 70.76%; accent-folded names added 328 links for 53,138 more pairs. On a fixed 2,000-query validation sample, the combined union had 430,900 candidates and 6,418 reachable true links, with a 0.9678 oracle ceiling. An oracle is a candidate upper bound, not model performance.

On all 22,133 held-out development queries, the broad union contained 4,743,229 pairs and 70,741 reachable positive links. The top-40 learned block retained 864,450 pairs and **70,735 of those 70,741** links, leaving the candidate oracle at **0.967876** versus 0.96793 for the broad union. Under the earlier 20,000-query Unicode matcher, top 40 lost one true selected link and no false selected links; raw macro F0.5 changed only 0.908276→0.908274. Top 45 added 106,709 candidates but recovered no further selection for that earlier matcher. The final larger matcher was evaluated directly on this exact top-40 candidate set. On each of two separate 2,000-query development and validation samples, the top-40 block kept every broad-pool positive. This ranking is measured, not an arbitrary truncation: a top-20 name/top-20 address lexical quota lowered validation oracle F0.5 to 0.939858.

**Final test last-stage candidate pairs:** [Fill after compact run and validator] across 1,732,544 test Source 1 records

## 4. Matching Model

We use two locally trained LightGBM 4.5.0 classifiers (MIT license, locally generated weights, no pretrained model or remote API). The preliminary 23-feature model was trained on a seeded 5,000-query sample and only ranks broad candidates for the top-40 block. The final 35-feature matcher was trained using a seeded 200,000-query sample, of which 190,086 Source 1 queries belonged to the training partition. Its core retrieval produced 38,808,759 candidate pairs and 600,773 reachable positive pairs. An earlier 20,000-query Unicode model mined the 20 hardest nonmatches and up to five random nonmatches per query; all reachable positives were retained. The resulting 5,246,815-row training set used an internal calibration fold and up to 1,200 LightGBM trees. Held-out-owned targets were excluded from training negatives.

The first 27 local features cover Unicode-normalized name and address ratios, token-sort and token-set similarity, accent-folded core-name similarity, exact-name indicators, token intersection and containment, address-number overlap/disagreement, field lengths, missing address, Source 3 indicator, and four first-address-number features. Eight added features compare locally ASCII-transliterated names and addresses, including token similarity and containment, while preserving original-script evidence. The pinned ISC-licensed `anyascii` text transform runs locally; it neither calls a remote service nor looks up business identities. Thresholds **0.65 for India and 0.75 for other countries** were selected on the separate 2,000-query development sample. France uses the other-country threshold without any country-specific feature or French training labels.

The output pass limits each predicted group to its 11 highest pair probabilities. This bound is the maximum true group size across all 2.2 million training Source 1 rows; it did not change any of the fixed 2,000 validation predictions. It is a conservative response to unlabeled French generic names that otherwise attract hundreds of model-selected records at different streets. Each target is assigned to at most one Source 1, choosing the highest model score when predictions compete. Every target had exactly one true owner in the supplied training labels. A fixed US house-number veto was tested but reduced this model's labeled score, so it is not applied. French quality is not directly measurable.

## 5. Results and Error Analysis

The final 35-feature hard-negative matcher with development-selected thresholds scored **0.910586 macro F0.5** on the fixed 2,000-query validation sample before group postprocessing, versus **0.904159** for the earlier 20,000-query Unicode matcher. India improved 0.865792→0.877439, US 0.933402→0.936385, and cross-script positive recall 0.8859→0.9305. The top-40 probe retained all broad-pool positive links on this sample; the new matcher was not separately scored behind top-40 on that sample. These are local results, not leaderboard scores. No French labeled F0.5 is available.

On the complete 22,133-query development partition, the final matcher achieved **0.918651 macro F0.5 after the top-11 cap and exclusive-owner passes on the exact top-40 candidate set**, with 64,387 true positive links, 1,516 false positive links, and 12,085 missed true links. The US score was 0.937644 and India 0.890004. The earlier 20,000-query Unicode matcher scored 0.908314 after these passes on the broad pool, with 62,261 true and 1,265 false links. The larger training sample therefore gained 2,126 true links at a cost of 251 extra false links while the top-40 block held candidate volume fixed. These development figures informed diagnosis and are not an independent leaderboard estimate.

False positives cluster around nearly identical business names at different units or streets and shared buildings. False negatives involve aliases, Indian script changes, blank/short addresses, number corruption, and positive pairs that never entered the candidate set. A blanket house-number mismatch rejection loses many true links, so number differences are learned rather than hard-vetoed. The larger model accepted more true and false links under development-selected thresholds; the macro F0.5 tradeoff improved on both labeled partitions. An earlier unlabeled first-20,000-test-query audit of the smaller Unicode model found a French generic-name tail before group capping, including 240/2,981 French groups above 11 links. Capping and exclusive ownership left none above 11, but that model's final French link count was 11,858 versus 9,398 in the uploaded baseline on this slice. Those counts do not measure the final larger model's French accuracy; public feedback remains necessary.

## 6. Conclusion

The method combines complementary fields and script-tolerant routes while controlling candidate volume and false merges. Its measured local result is 0.910586 macro F0.5 on 2,000 held-out validation queries with development-selected thresholds, and 0.918651 on the full development partition with exact top-40 candidates and group postprocessing. The final ZIP size and public score are separate checks. The main uncertainty is generalization to France and residual alias/candidate-miss cases, which were not covered by French labels or external data.

## Appendix: Code Artifacts

`code/business_entity_resolution/src/build_index.py` builds a local target index, `src/infer.py` creates the broad intermediate, and `src/two_stage_rescore.py` applies bundled `model.joblib` as candidate ranker before passing its recorded top-40 set to `generalized_model.joblib`. `src/cap_predictions.py` and `src/resolve_exclusivity.py` apply final group decisions. `src/train_hard_negative.py`, bundled `generalized_seed_model.joblib`, and `src/validation.py` document final model reconstruction and the deterministic split. Exact commands and pinned dependencies are in `code/business_entity_resolution/README.md` and `requirements.txt`. We used no external identity lookup, registry, geocoding, or remote model inference.
