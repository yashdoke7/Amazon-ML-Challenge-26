# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Vulcans
**Team Members:** Yash Kailas Doke, Harsh Jitendra Jain, Ayush Tiwari, Vedant Kaulgekar
**Submission Date:** 27 September 2026

## 1. Executive Summary

We resolve each deduplicated Source 1 business to zero or more Source 2/3 records using a country-aware candidate union and a locally trained LightGBM pair classifier. Separate name and address retrieval quotas protect records with missing fields, aliases, Indian script changes, and French accents. The final model learns first-address-number agreement, then applies group-size and single-owner decisions to contain collisions.

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

We trained a LightGBM 4.5.0 classifier (MIT license, locally generated weights, no pretrained model or remote API) on a seeded 20,000-query sample from the supplied training data. Its complete core-route training extract had 3,948,642 pairs, including 63,398 positive candidates. The training split excluded negative pairs whose targets belonged to development or validation entities.

The 27 local features cover Unicode-normalized name and address ratios, token-sort and token-set similarity, accent-folded core-name similarity, exact-name indicators, token intersection and containment, address-number overlap/disagreement, field lengths, missing address, Source 3 indicator, and four features for the first address number (equality, capped distance, nearby mismatch, and presence). We selected a global score threshold of 0.75 on a separate 2,000-query development sample. An India-only 0.65 threshold improved that sample and a frozen 2,000-query validation sample; US and unlabeled France remain at 0.75.

The output pass limits each predicted group to its 11 highest pair probabilities. This bound is the maximum true group size across all 2.2 million training Source 1 rows; it did not change any of the fixed 2,000 validation predictions. It is a conservative response to unlabeled French generic names that otherwise attract hundreds of model-selected records at different streets. Each target is assigned to at most one Source 1, choosing the highest model score when predictions compete. Every target had exactly one true owner in the supplied training labels. A fixed US house-number veto was tested but reduced this model's labeled score, so it is not applied. French quality is not directly measurable.

## 5. Results and Error Analysis

With combined candidates, the frozen 27-feature model and global threshold 0.75 scored **0.89091 macro F0.5** on the fixed 2,000-query validation sample (5,542 true positive links, 188 false positive links, 1,353 missed true links). The India-only 0.65 threshold selected on development raised this frozen validation score to **0.89272**. These are local sample results, not leaderboard scores. No French labeled F0.5 is available.

On the complete 22,133-query development partition, the same combined candidate route and frozen matcher achieved **0.89761 macro F0.5 after the top-11 cap and exclusive-owner passes**, with 62,584 true positive links, 2,495 false positive links, and 13,888 missed true links. The US score was 0.93056 and India 0.84791. A uniform 0.75 threshold scored 0.89574 after those passes; the original 23-feature baseline with its US veto scored 0.88331. These development figures informed diagnosis and are not an independent leaderboard estimate.

False positives cluster around nearly identical business names at different units or streets and shared buildings. False negatives involve aliases, Indian script changes, blank/short addresses, number corruption, and positive pairs that never entered the candidate set. A blanket house-number mismatch rejection loses many true links, so number differences are learned rather than hard-vetoed. Lowering India's threshold increases recall but also adds false links and reduces singleton accuracy.

## 6. Conclusion

The method combines complementary fields and script-tolerant routes while controlling candidate volume and false merges. Its measured local result is 0.89272 macro F0.5 on 2,000 held-out validation queries with the India threshold override. The main uncertainty is generalization to France and residual India cross-script/alias cases, which were not covered by French labels or external data.

## Appendix: Code Artifacts

`code/business_entity_resolution/src/build_index.py` builds a local target index, `src/infer.py` creates the candidate TSV, and `src/rescore_candidates.py` applies bundled `number_model.joblib` to that saved candidate set with the country threshold override. `src/cap_predictions.py` and `src/resolve_exclusivity.py` apply final group decisions. `src/train.py` and `src/validation.py` document model reconstruction and the deterministic split. Exact commands and pinned dependencies are in `code/business_entity_resolution/README.md` and `requirements.txt`. We used no external identity lookup, registry, geocoding, or remote model inference.
