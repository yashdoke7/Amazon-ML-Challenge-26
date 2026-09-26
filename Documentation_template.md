# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Registered team name]  
**Team Members:** [Registered member names]  
**Submission Date:** 27 September 2026

## 1. Executive Summary

We resolve each deduplicated Source 1 business to zero or more Source 2/3 records using a country-aware candidate union and a locally trained LightGBM pair classifier. Separate name and address retrieval quotas protect records with missing fields, aliases, Indian script changes, and French accents. A bounded final group-size pass addresses generic-name collisions observed in unlabeled French test records.

## 2. Methodology

### 2.1 Problem Analysis

Training contains 2,206,821 Source 1 queries, 10,320,219 Source 2/3 targets, and 7,638,365 positive links. Test adds France, which has no labeled training examples. In a labeled pair sample, 11.2% had weak name similarity, 7.9% weak address similarity, 4.4% blank target address, and 7.1% changed name script. Identical names and addresses can also belong to distinct labeled entities. A whole-string equality rule or a single name/address similarity score therefore loses true links and creates false merges. The metric is macro F0.5 per Source 1, including empty true groups; we optimized for complete group quality and precision, not pair accuracy.

### 2.2 Solution Strategy

**Approach Type:** Multiple blocking routes plus a pair classifier and bounded group postprocessing.  
**Core contribution:** Separate rare-token retrieval and field-specific top-100 ranking, combined with compact/accent-folded name equality and targeted Indian address-token overlap. All routes use equal country as an open-set block, including France.

## 3. Candidate Generation

We tokenize Unicode letters/digits in names and addresses and choose each query's three least frequent target tokens with document frequency at most 3,000 and length at least four. The union of their postings is ranked independently by name and address Jaro-Winkler similarity; a pair survives if either rank is at most 100. We add compact names after common legal-suffix removal, accent-folded compact names, and, for Indian queries, the top 50 candidates by address-token overlap when at least two tokens overlap. Country must match. The final `candidate_pairs.tsv` lists this exact union before classifier scoring.

On all 22,133 held-out development Source 1 records, the core route retrieved 91.18% of positive links with 4,518,733 candidate pairs and an oracle macro-F0.5 ceiling of 0.9617. Independently, India address-overlap top 50 added 686 true links for 171,364 more pairs and raised cross-script recall from 62.53% to 70.76%; accent-folded names added 328 links for 53,138 more pairs. On a fixed 2,000-query validation sample, the combined union had 430,900 candidates and 6,418 reachable true links, with a 0.9678 oracle ceiling. An oracle is a candidate upper bound, not model performance.

**Final test candidate pairs:** [Fill after complete run and validator]

## 4. Matching Model

We trained a LightGBM 4.5.0 classifier (MIT license, locally generated weights, no pretrained model or remote API) on a seeded 5,000-query sample from the supplied training data. Its complete core-route training extract had 991,451 pairs, including 15,738 positive candidates. The training split excluded negative pairs whose targets belonged to development or validation entities.

The 23 local features cover Unicode-normalized name and address ratios, token-sort and token-set similarity, accent-folded core-name similarity, exact-name indicators, token intersection and containment, address-number overlap/disagreement, field lengths, missing address, and Source 3 indicator. The frozen threshold is 0.65, selected through the internal training calibration; a later 2,000-development-query threshold sweep favored 0.775 but failed to improve the unchanged validation sample, so we retained 0.65.

The output pass limits each predicted group to its 11 highest pair probabilities. This bound is the maximum true group size across all 2.2 million training Source 1 rows; it did not change any of the fixed 2,000 validation predictions. It is a conservative response to unlabeled French generic names that otherwise attract hundreds of model-selected records at different streets. French quality is not directly measurable.

## 5. Results and Error Analysis

With combined candidates and the frozen core-trained model at threshold 0.65, the fixed 2,000-query validation sample scored **0.8754 macro F0.5**, with 5,521 true positive links, 306 false positive links, 1,374 missed true links, and 74.8% singleton accuracy on 103 true singletons. US macro F0.5 was approximately 0.906 and India approximately 0.827 in the earlier core-candidate run; these are local sample results, not leaderboard scores. The alternative combined-trained model scored 0.8730 on the same sample. No French labeled F0.5 is available.

False positives cluster around nearly identical business names at different units or streets and shared buildings. False negatives involve aliases, Indian script changes, blank/short addresses, number corruption, and positive pairs that never entered the candidate set. Exact house-number disagreement was not used as an automatic rejection because supplied positives also contain number corruption.

## 6. Conclusion

The method combines complementary fields and script-tolerant routes while controlling candidate volume and false merges. Its measured local result is 0.8754 macro F0.5 on 2,000 held-out validation queries. The main uncertainty is generalization to France and residual India cross-script/alias cases, which were not covered by French labels or external data.

## Appendix: Code Artifacts

`code/business_entity_resolution/src/build_index.py` builds a local target index, `src/infer.py` generates both output TSVs from the supplied test files and bundled `model.joblib`, and `src/cap_predictions.py` applies the final group bound. `src/train.py` and `src/validation.py` document model reconstruction and the deterministic split. Exact commands and pinned dependencies are in `code/business_entity_resolution/README.md` and `requirements.txt`. We used no external identity lookup, registry, geocoding, or remote model inference.
