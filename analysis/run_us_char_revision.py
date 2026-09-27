"""Verify and package the frozen US character-address retrieval revision.

Never replaces output/final until every validator and ZIP-size check passes.
"""

import csv
import json
import re
import shutil

from run_us_address_revision import (DATA, DB, EXPECTED, FINAL, OUTPUT,
                                     PACKAGE, ROOT, candidate_count, row_count,
                                     run, sha256)


EXPECTED_US = 663_106
BACKUP = OUTPUT / "final_us_both_verified"


def verify_extra(extra):
    if row_count(extra) != EXPECTED_US:
        raise ValueError("US character-address output is incomplete")
    source1 = DATA / "test_source1.tsv"
    with source1.open(encoding="utf-8", newline="") as source, \
         extra.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Unexpected US character-address output header")
        count = 0
        for query in csv.DictReader(source, delimiter="\t"):
            if query["country"] != "US":
                continue
            row = next(reader, None)
            if row is None or row["source1_entity_id"] != query["entity_id"]:
                raise ValueError(f"US retrieval row/order mismatch at {count}")
            ids = row["candidate_entity_ids"].split(",") if row["candidate_entity_ids"] else []
            if len(ids) > 10 or len(ids) != len(set(ids)):
                raise ValueError(f"Invalid US retrieval quota at {query['entity_id']}")
            count += 1
        if count != EXPECTED_US or next(reader, None) is not None:
            raise ValueError("US retrieval count mismatch")
    print("verified US char retrieval rows", count, flush=True)


def main():
    if not (OUTPUT / "US_BOTH_REVISION_COMPLETE.json").is_file():
        raise ValueError("Expected verified US address-plus-name base")
    base_candidate = OUTPUT / "generalized_us_both_candidate_pairs.tsv"
    base_raw = OUTPUT / "generalized_us_both_results_uncapped.tsv"
    extra = OUTPUT / "us_char_address_top10.tsv"
    candidate = OUTPUT / "generalized_us_char_candidate_pairs.tsv"
    raw = OUTPUT / "generalized_us_char_results_uncapped.tsv"
    capped = OUTPUT / "generalized_us_char_results_capped.tsv"
    matching = OUTPUT / "generalized_us_char_results.tsv"
    doc = ROOT / "docs/Documentation_generalized_us_char.md"
    archive = OUTPUT / "Vulcans_us_char_submission.zip"
    pdf = OUTPUT / "pdf/Vulcans_approach_summary_char.pdf"
    if any(row_count(path) != EXPECTED for path in (base_candidate, base_raw)):
        raise ValueError("Verified base candidate/raw files are incomplete")
    verify_extra(extra)
    run(PACKAGE / "src/merge_address_candidates.py", "--data-dir", DATA,
        "--db", DB, "--base-candidate", base_candidate,
        "--base-matching", base_raw, "--extra", extra,
        "--candidate", candidate, "--matching", raw,
        "--model", PACKAGE / "generalized_model.joblib",
        "--blank-specialist", PACKAGE / "blank_frequency_model.joblib",
        "--country", "US", "--general-threshold", "0.75",
        "--blank-threshold", "0.8", "--sparse-extra", "--workers", "4")
    if row_count(candidate) != EXPECTED or row_count(raw) != EXPECTED:
        raise ValueError("Merged candidate/raw rows incomplete")
    run(PACKAGE / "src/cap_predictions.py", "--data-dir", DATA,
        "--db", DB, "--model", PACKAGE / "generalized_model.joblib",
        "--input", raw, "--output", capped, "--cap", "11")
    run(PACKAGE / "src/resolve_exclusivity.py", "--data-dir", DATA,
        "--db", DB, "--model", PACKAGE / "generalized_model.joblib",
        "--input", capped, "--output", matching)
    count = candidate_count(candidate)
    content = doc.read_text(encoding="utf-8")
    updated, replaced = re.subn(
        r"(\*\*Final test last-stage candidate pairs:\*\* )[^\n]+?( across 1,732,544 test Source 1 records)",
        rf"\g<1>{count:,}\g<2>", content)
    if replaced != 1:
        raise ValueError("Cannot update methodology candidate count")
    doc.write_text(updated, encoding="utf-8")
    if not pdf.is_file():
        raise ValueError("New approach PDF missing")
    run(ROOT / "analysis/package_submission.py", "--team-name", "Vulcans",
        "--check-ids", "--variant", "compact_generalized",
        "--matching-path", matching, "--candidate-path", candidate,
        "--doc-path", doc, "--archive-path", archive,
        "--compresslevel", "9")
    size = archive.stat().st_size
    if size >= 512_000_000:
        raise ValueError(f"Archive exceeds portal limit: {size}")
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in ("matching_results.tsv", "Vulcans_submission.zip",
                 "Vulcans_approach_summary.pdf"):
        old = FINAL / name
        saved = BACKUP / name
        if not saved.exists():
            shutil.copy2(old, saved)
    shutil.copy2(matching, FINAL / "matching_results.tsv")
    shutil.copy2(archive, FINAL / "Vulcans_submission.zip")
    shutil.copy2(pdf, FINAL / "Vulcans_approach_summary.pdf")
    result = {"queries": EXPECTED, "us_char_queries": EXPECTED_US,
              "last_stage_candidate_pairs": count, "archive_bytes": size,
              "matching_sha256": sha256(matching),
              "candidate_sha256": sha256(candidate),
              "archive_sha256": sha256(archive),
              "final_dir": str(FINAL), "fallback_dir": str(BACKUP)}
    (OUTPUT / "US_CHAR_REVISION_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
