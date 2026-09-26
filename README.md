# Amazon ML Challenge 2026: Business Entity Resolution

Private team workspace for analysis and a reproducible submission. Source 1 is the deduplicated reference; the task is to return every matching Source 2/3 ID for each test Source 1 ID, including empty lists for singletons. The private leaderboard scores macro F0.5 per Source 1 entity.

## Current stage

Row-level exploratory analysis and candidate-retrieval probes are complete. A first matching baseline has been evaluated on a 2,000-entity validation sample; full-test inference and leaderboard submission are **not yet complete**. The two canonical documents are [findings and decisions](docs/FINDINGS.md) and [next experiments](docs/NEXT.md). Supporting measurements are in [the detailed EDA](analysis/DEEP_EDA.md), [the retrieval probes](analysis/RETRIEVAL_PROBES.md), and [the experiment log](docs/EXPERIMENT_LOG.md).

## Layout

```text
analysis/                     reproducible EDA scripts, findings, and chart
docs/                         handoff, rules, risk matrix, experiment log
code/business_entity_resolution/   final runnable pipeline (in progress)
output/                       eventual TSV predictions (ignored by Git)
6ab10eb3b23ba_student_resource/   supplied challenge material (ignored by Git)
```

The challenge dataset and extracted raw-record JSON outputs are deliberately excluded from Git. Keep the supplied `student_resource` directory at its original path when running the existing EDA scripts. The final pipeline will accept a configurable dataset path.

## Reproducing current evidence

Use Python 3.11 and install `analysis/requirements.txt`. From the repository root:

```powershell
python analysis/deep_eda.py
python analysis/collision_eda.py
python analysis/audit_labels.py
python analysis/frequency_eda.py
python analysis/source_patterns.py
python analysis/plot_eda.py
```

The original problem statement and validator are under `6ab10eb3b23ba_student_resource/student_resource/`. We use them locally but do not publish the supplied files.

## Competition guardrails

No external business-identity lookup, geocoding API, external data augmentation, or remote matching service is permitted. Models must satisfy the stated MIT/Apache 2.0 and at-most-8B-parameter condition. See [rules and submission checklist](docs/RULES.md). The dataset may be used locally for training and inference; it is not committed to this repository.
