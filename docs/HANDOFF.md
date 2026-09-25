# Handoff checkpoint

_Updated 25 September 2026. Read this first when resuming the project or switching assistants._

## Objective and status

Target: maximize private-leaderboard macro F0.5 on Source 1 → all matching Source 2/3 records while meeting the complete submission and fair-play rules in [RULES.md](RULES.md). The intended competition outcome is top 50 for the Applied Scientist Intern interview, with top 10/final prizes as stretch goals.

**Completed:** inspected the supplied challenge PDF and transcript; audited full dataset sizes and label integrity; sampled and reviewed true match groups, hard exact-key nonmatches, Indian script changes, French test text, corruption, shared addresses, and source-specific noise. Ran exact normalized candidate-key probes against the full target universe. See [DEEP_EDA.md](../analysis/DEEP_EDA.md), [RETRIEVAL_PROBES.md](../analysis/RETRIEVAL_PROBES.md), [INITIAL_STRATEGY.md](../analysis/INITIAL_STRATEGY.md), and [eda_patterns.png](../analysis/eda_patterns.png). The EDA scripts are reproducible; generated JSON with extracted rows is local-only and ignored by Git.

**Not completed:** candidate-recall benchmark, validation split, matching model, full-test predictions, portal upload, final archive. Do not state a model score yet.

## Essential measured facts

- Train: 2,206,821 S1; 5,034,616 S2; 5,285,603 S3; 7,638,365 true links. Test: 1,732,544 S1 and 9,969,589 combined targets. Test includes 259,452 French S1 without French labels.
- 5.58% of train S1 rows are singletons. Mean true link count is 3.46; 11.4% have more than five true links.
- Full label audit: every target exists, every match stays within country, and each target is owned by only one S1.
- 30k random S1 sample (103,685 true links): 11.2% of positives have weak name similarity (<50), 7.9% weak address similarity, 0.94% both; 4.4% target addresses are blank; 7.1% change name script. For India/S2, 22.7% change name script.
- 5k S1 collision sample: case-insensitive exact-name retrieval finds 30,495 candidates, only 1,908 labeled true; exact-address retrieval finds 1,569 candidates, 1,255 labeled true. Their union covers only 17.6% of the sample's 17,312 true links. Generic names can create hundreds of candidates. Exact addresses can host multiple businesses.
- On the same 5k sample, Unicode/punctuation normalized name/address union covers 29.5% of links. A crude legal-suffix-reduced name plus normalized address covers 47.6%, but yields 183,461 candidate pairs and a maximum block of 1,331 targets. Fuzzy retrieval is still required.
- Source 3 includes far more `DBA`/`aka`/`t/a` name variants than Source 2. France contains accented names and addresses; five-digit postal numbers are rare in the supplied addresses.

## Current decisions

1. Keep raw name/address text. Create parallel normalized views; do not overwrite information such as house/unit numbers, scripts, accents, legal suffixes, or alias substrings.
2. Candidate generation must use complementary **name-led and address-led** routes. Exact equality is only a small seed; measure recall and candidate volume before a matcher.
3. Use open-set country equality as a blocking key: all train links agree on country, while France must pass through the same code path.
4. Treat source-specific noise, cross-script names, aliases/domains, missing addresses, shared locations, near-identical labeled nonmatches, and singletons as explicit validation slices.
5. Do not use external identity lookup or remote inference. Check model-weight license and parameter count before use.

## Immediate next work

Continue the candidate-generation benchmark **before** training the matcher. Freeze a deterministic S1 entity split with all their S2/S3 positives grouped by owner. Next test token/character retrieval with explicit block-volume estimates. For each retrieval route report: positive-edge recall, fraction of S1 with every true link present, candidate pairs per S1, runtime, and recall for the risk slices above. Avoid leakage of held-out positives into training negatives. Record each experiment in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), then decide which routes merit full-scale indexing.

## Repository and data paths

- Working folder: `C:\Users\ASUS\Desktop\CS\Projects\Amazon ML Challenge`
- Supplied data: `6ab10eb3b23ba_student_resource/student_resource/dataset/` (ignored by Git)
- Supplied validator: `6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py`
- Private remote: `https://github.com/yashdoke7/Amazon-ML-Challenge-26`

Do not commit the supplied dataset, large predictions, model binaries, credentials, or raw-record JSON extracts. Scripts and aggregate analysis can be committed. Check `git status` before pushing.
