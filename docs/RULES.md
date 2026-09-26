# Challenge rules and submission contract

Source: supplied 2026 problem-statement PDF, local `student_resource/README.md`, and validator. The [Unstop challenge page](https://api.unstop.com/hackathons/amazon-ml-challenge-2026-amazon-1743604) supplies the schedule and rewards. If the live portal imposes an additional upload constraint, verify it before final submission.

## Prediction target

- Source 1 is a deduplicated reference. Predict all matching Source 2 and Source 3 entity IDs for **every** test Source 1 ID; the list may be empty.
- Source files provide `entity_id`, `business_name`, `business_address`, and `country`. IDs identify records, not cross-source entities. Country labels are open-set: train contains US and India; test adds France.
- Ground truth gives one row per train Source 1, with a comma-separated list of true target IDs.
- Score is macro F0.5 over Source 1 rows, including singletons. For a true singleton, an empty prediction scores 1 and any nonempty prediction scores 0. Use the exact formula in the PDF, not pairwise accuracy or micro F-score, to select thresholds.
- Public leaderboard uses a subset of test; private leaderboard determines final rankings.

## Live upload

`output/matching_results.tsv` is UTF-8, **tab-separated**, with exactly these columns:

```text
source1_entity_id<TAB>matched_entity_ids
```

It must have exactly one row for each test Source 1 ID, including empty second fields for singletons. IDs in `matched_entity_ids` are comma-separated existing test S2-/S3- IDs only. No duplicate row IDs or duplicate IDs inside a list.

## Final archive

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

`candidate_pairs.tsv` has exactly `source1_entity_id<TAB>candidate_entity_ids`, one row per test Source 1. It must list the **last candidate set actually passed to the matching model**, after earlier blocking or filtering. Every final matched ID must appear in that row's candidate list. It is audited rather than leaderboard-scored.

The code copy must regenerate both output files end to end from supplied train/test TSVs, with pinned dependencies and exact instructions. Fill the supplied methodology template with problem analysis, blocking, features/model, threshold selection, validation, error analysis, and results. The challenge-specific PDF says there is no page limit. The general Unstop page says 1–2 pages; check the portal for a hard file/page limit.

The team reports that the live portal caps the final ZIP upload at **512 MB**. The original 353.9-million-pair baseline ZIP is 2.02 GB and therefore cannot be submitted there. A compact last-stage candidate set must be generated and validated before the final archive is used; do not merely omit the required candidate file. This limit came from the team's portal observation, not the supplied PDF.

Run the supplied `utils/validate_submission.py` against both outputs and `dataset/test`. Use `--check-ids` if memory permits; it performs the optional target-existence check. Also independently verify the output row set, IDs, candidate subset, and reproducibility.

## Fair play and model constraints

- **No external data lookup** to resolve any supplied business. This includes business registries, directories, commercial ER services, maps/geocoding APIs, and internet data augmentation. Do not send record text to a remote model/API. Our method research is separate from inference.
- Use the provided data for features, training, blocking, validation, and test inference. Treat unlabeled test use beyond inference cautiously and document any transductive step.
- The final model must have an MIT or Apache 2.0 license and at most 8 billion parameters. Verify each pretrained model's exact repository license and keep a local license record before adoption. Do not assume a library license settles the model-weights license.
- Keep the supplied dataset out of GitHub. The private repository contains code, derived aggregate findings, documentation, and reproducible procedures.
