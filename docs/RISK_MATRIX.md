# Observed failure modes and tests

These are hypotheses to validate, not accepted matching rules. A mitigation moves forward only after measured recall and macro F0.5 improve on the held-out split.

| Failure mode | Evidence from supplied data | Candidate/feature hypothesis | Check before adoption |
| --- | --- | --- | --- |
| Cross-script Indian name | 22.7% of sampled India/S2 positives change name script | Address-led retrieval, Unicode-aware tokens, optional local transliteration view | Recall on script-change links; false merges at shared addresses |
| French accents and abbreviations | French names/addresses often contain non-ASCII Latin letters; no French labels | Raw plus accent-folded Latin views; street abbreviation features | Country holdouts and French candidate-volume inspection |
| Unseen language beyond train | Train labels cover US/India only | Unicode token/character retrieval with field-level fallback; optional licensed multilingual reranker on residuals | Cannot claim equal quality without labels; stress-test scripts held out from training |
| Target address missing | 4.4% of sampled positives; high token union retrieves 61.0%, but joint Jaro-Winkler top100 only 5.0% | Name-led route with a reserved quota; explicit missingness feature | Recall and false merges among short/generic names |
| Transliterated or alias name | Indian scripts, `DBA`/`aka`, domains, occasionally unrelated trade names | Parallel alias/domain-stem features; S2/S3 sibling evidence | False-positive rate for same-address businesses |
| Generic repeated name | `Meridian` yields 366 exact-name candidates in sampled queries | Rare-term weighting, address/locality disambiguation, block-size budget | p99 candidate count and recall for generic names |
| Shared exact address | Up to 14 S1 businesses at one address; labeled wrong targets exist | Require name/alias evidence or graph agreement; abstain on ambiguity | Precision on co-located businesses |
| Near-identical wrong record | `FRG` vs `FRGU` at same address is a labeled nonmatch | Hard-negative mining and confidence margin | Error slice, possible label ambiguity; no unvalidated hard rule |
| Address number corruption | 7.3% positive pairs have disjoint number sets | Soft number similarity and component-aware edits | Recall loss from strict number filters |
| Long or reordered addresses | India S1 averages 77.7 chars, S3 61.0; components reorder | Token/character retrieval and rare address terms | Recall by address length and source |
| Literal placeholder vs place name | `NULL` component exists; `Null Bazar` is legitimate text | Remove only isolated placeholders in a derived view | Manual samples and collision change |
| Singleton false merge | 5.58% of S1 truth rows empty; any predicted link scores 0 there | Calibrated reject option and macro-F0.5 threshold | Singleton accuracy and full S1-set score |
| Multiple true targets | 11.4% of S1 have >5 matches | No tiny final top-k cap; candidate recall over complete sets | Complete-set recall by match count |
| Target claimed by multiple S1 | Full train truth has unique target ownership | Resolve only conflicts with calibrated edge scores | Score delta and ambiguity audit |
| Public/private shift | France appears only in test; public is a subset | Local fixed holdout, source/country slices, limited LB tuning | Public score versus robust validation and France coverage |
| Resource limits at 10M targets | Full test cross-product is 17.27 trillion; sampled high token union extrapolates ~3.24B pairs | Multi-pass blocking, pre-count join volumes, separate name/address quotas, batch/disk-backed features | Runtime, memory, pair budget before full inference |
