# Amazon ML Challenge 2026: initial evidence and plan

_25 September 2026, about 17.5 hours into the 72-hour round. This is a working research note, not the final methodology submission._

For the subsequent row-level study, examples, cleaning implications, and hard-negative measurements, see [DEEP_EDA.md](DEEP_EDA.md). Its similarity percentages use a larger random sample and a documented Unicode normalization, so they need not equal this initial raw-text sample's percentages.

## Objective and contract

Match every test Source 1 business to **all** records of the same business in Source 2 and Source 3, or output an empty list. Source 1 is deduplicated; Source 2/3 can contain multiple noisy variants. The scored `matching_results.tsv` needs exactly one tab-separated row per test Source 1 ID. The final archive also needs the actual pre-model `candidate_pairs.tsv`, reproducible code, pinned environment, and methodology document. `utils/validate_submission.py` checks the output format. The score is **macro F0.5 per Source 1**, including singleton rows, and the private leaderboard determines final ranking. External business-identity lookup or augmentation is prohibited. The final model must satisfy the MIT/Apache 2.0 and <=8B-parameter rule.

The [official Unstop page](https://api.unstop.com/hackathons/amazon-ml-challenge-2026-amazon-1743604) lists a 25–27 September 2026 window, top-50 Applied Scientist Intern PPIs, top-10 finalist presentations, and cash prizes for the top three. The challenge-specific problem-statement PDF confirms the local README's **no-page-limit** methodology instruction; the general Unstop page says 1–2 pages. Follow the challenge-specific PDF for content and check whether the active portal imposes a hard upload limit before packaging.

## Initial mental model

Think of each S1 record as a **reference business dossier**. S2 and S3 are vendor fragments. We must decide which fragments belong in that dossier, including the possibility that none do. This is a retrieval-and-verification problem with an abstain option:

```text
S1 business -> independent name/address candidate routes -> bounded plausible set
            -> field-aware pair scores -> accept/reject each link
            -> resolve conflicting target ownership -> full match list or empty
```

The first stage must find true links even when one field is badly corrupted. The second must distinguish a true alias from a similarly named branch or a different business at the same address. The scoring unit is the **whole S1 dossier**, so evaluate complete predicted ID sets and singletons, not just pairwise accuracy. S2–S3 relationships can supply extra evidence but are not themselves a requested output. The video transcript supplied by the user confirms this two-stage framing and says that name and address are the primary identity fields; `country` is an additional provided field and must remain open-set.

Use the PDF's exact F0.5 formula in code. Its verbal “precision twice as heavily” explanation is directional; threshold selection must optimize the defined formula over complete S1 match sets, including empty sets.

## Dataset facts measured locally

| Split | Source 1 | Source 2 | Source 3 | Countries |
| --- | ---: | ---: | ---: | --- |
| Train | 2,206,821 | 5,034,616 | 5,285,603 | US, India |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | US, India, France |

Test Source 1 includes 663,106 US, 809,986 India, and **259,452 France** rows. A full test cross-product would be about **17.27 trillion pairs**, so candidate generation is mandatory. Source 1 has no missing names or addresses. Source 2/3 addresses are missing on 168,967/175,916 train rows and 129,408/136,098 test rows, respectively.

Training truth has 7,638,365 positive links, an average of 3.46 per Source 1. There are 123,247 singleton Source 1 rows (5.58%). Each positive Source 2/3 ID appears under exactly one Source 1 ID in all training labels. About 73.4% of train Source 2 and 74.6% of train Source 3 records are labeled matches; the others act as distractors. Train singleton rates are nearly identical in the US and India (about 5.6%). No Source 1 record has identical name, address, and country text across train and test.

On the first 20,000 training truth rows (69,144 positive pairs), case-insensitive exact name occurs in 10.8% and exact address in 7.2%. The target address is absent for 4.4% of these positives. Token-sorted name similarity falls below 50/100 for 12.7%, address similarity for 7.9%, and **both** for only 1.0%. S2 has more low-name positives (14.3% versus S3's 11.3%); S3 has more low-address positives (9.9% versus S2's 5.8%). These are sample statistics, not a candidate-recall measurement. The data contains non-Latin name transliterations, domain-style names, DBA aliases, reordered addresses, typos, and literal `null` fragments. One true pair has `Maure Williams Colombier Inc` versus `Dréxkor` with a strongly similar address; another has English versus Tamil name text with essentially identical addresses.

## What past winners actually teach us

The [2024 winning team](https://github.com/nachiketashunya/Amazon-ML-Challenge-2024) reported an F1 jump from 0.679 to 0.865 after a second fine-tune on 1,600 carefully corrected examples. The [2024 rank-6 team](https://github.com/arnav10goel/Amazon-ML-Challenge-24) emphasized error-driven examples and output cleanup. A [2025 private rank-5 team](https://github.com/RudrakshSJoshi/amlc-multimodal-mlp) combined complementary feature streams, while the [2023 winner](https://github.com/pj-mathematician/Amazon-ML-Challenge-2023/blob/main/amazon-ml-challenge-2023-winner-solution.ipynb) used nearest-neighbor retrieval followed by LightGBM. Those were different tasks; the transferable lesson is **measured retrieval, clean validation, error taxonomy, and complementary evidence**, not their exact architectures.

## Working goals, in order

1. **Secure a valid submission early.** Build a fast name/address retrieval baseline, pair scorer, exact macro-F0.5 evaluator, both output files, and run the validator. Record a public leaderboard score without treating it as the only truth. The live portal takes only `matching_results.tsv`; the archive also requires `candidate_pairs.tsv` containing the exact pairs presented to the scorer, after any early filtering.
2. **Measure the recall ceiling.** On an isolated labeled validation set, report candidate recall overall and by source, country, missing-address, low-name-similarity, and match-count groups. Missing links at blocking cannot be repaired by the classifier.
3. **Improve precision with hard negatives.** Train a model on retrieved nonmatches, not only random negatives; score name and address separately, numeric address components, locality, legal suffixes, scripts, and source-specific patterns. Tune decisions on macro-F0.5 and inspect false merges and singleton errors.
4. **Stress-test generalization.** Train on US and validate on India, then reverse, as imperfect proxies for unseen France. Keep countries as open-set labels. Inspect unlabeled French text distributions without using external business databases.
5. **Test a distinctive graph layer.** Because positive target IDs are exclusive to one Source 1, resolve competing Source 1 claims globally. Explore high-confidence Source 2/3 neighborhoods that recover aliases reachable through another variant. Accept only changes that improve isolated validation and preserve candidate-file lineage.
6. **Freeze and package in time.** Leave a substantial final buffer for full-test inference, `--check-ids` validation if memory permits, documented reproduction, archive, and final upload. The PDF allows a detailed methodology document; check the portal's file or page limit before upload.

## Candidate experiments to run next

| Experiment | Testable claim | Measurement |
| --- | --- | --- |
| Separate name and address retrieval | Orthogonal views recover most positives, including aliases and transliterations | Recall@k by view and union, candidate volume, runtime |
| Character n-gram and token retrieval | Character n-grams handle typographic noise; tokens handle reordered addresses | Incremental recall over exact/normalized keys |
| Source and country aware pair features | S2/S3 and US/India have different noise channels | Macro-F0.5 and subgroup false merges |
| Hard-negative mining | Ambiguous neighboring businesses drive false positives | Precision and macro-F0.5, especially singletons |
| Global exclusive assignment | A target claimed by multiple S1 rows should usually have one owner | Conflict count and score delta |
| S2–S3 agreement and graph expansion | A name-only or address-only variant can reveal a harder sibling, while independent agreement can validate an ambiguous link | Recall gain versus precision loss and runtime; no unconditional transitive closure |

Primary-source method references support this order: [Splink's blocking guide](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html) recommends multiple selective passes and measuring comparison counts; its [term-frequency guide](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html) explains why common business-name terms are weak evidence. [Ditto](https://www.vldb.org/pvldb/vol14/p50-li.pdf) motivates field-aware matching and difficult training examples. [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) is MIT licensed; [CatBoost](https://github.com/catboost/catboost) is Apache 2.0. A multilingual encoder or reranker should be evaluated only on residual misses or ambiguous top candidates: 10 million 384-dimensional float32 vectors alone require about 14.3 GiB before indexing and record storage. [BGE-M3](https://huggingface.co/BAAI/bge-m3) is a possible MIT multilingual option, subject to runtime measurement. Avoid pretrained address parsers that incorporate outside geographical datasets unless the organizers explicitly allow them.

At top 50 candidates per test Source 1, the pipeline creates about 86.6 million pairs. Batch the candidate joins, store compact integer IDs and features on disk, and stream scoring; do not hold Python object rows for all pairs. Before each blocking join, estimate its comparison count as the sum of `S1_block_size × target_block_size` over keys.

## Validation design

Split by Source 1 entity, keeping all its labeled Source 2/3 records together; distribute unmatched Source 2/3 distractors into the same folds. Avoid using validation positives as training negatives. Evaluate blocking recall separately from final macro-F0.5. Keep a fixed seed and a hidden local holdout to limit tuning to the public leaderboard. Report per-country and error-type metrics as well as the aggregate.

## Scope of outside research

Prior solutions, papers, method documentation, and model licenses may guide algorithm design. Do not query external registrations, maps, business directories, geocoders, APIs, or services to resolve any supplied record. All features and predictions must come from the provided TSV files and permitted models/code.
