# Reproducible entity-resolution pipeline

This directory will be copied into the final submission archive. It is currently a scaffold; no inference entry point or scored model exists yet. Development proceeds in [the experiment log](../../docs/EXPERIMENT_LOG.md).

Required final contents:

- `src/`: all blocking, feature, matching, calibration, and output-generation code.
- `requirements.txt`: exact versions and model-weight/license references.
- This README: exact commands from supplied train/test TSVs to both final output files, hardware assumptions, runtime, and validation instructions.

The final pipeline must create `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The latter must list the exact last-stage candidates presented to the matching model. Do not add an entry point here until it is implemented and verified.
