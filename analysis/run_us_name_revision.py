"""Finish and package the measured US-name route after the verified US-address run.

The existing final submission remains untouched unless the new archive passes
the complete validators and the strict 512,000,000-byte check.
"""

import json
import re
import shutil
from pathlib import Path

from run_us_address_revision import (DATA, DB, EXPECTED, FINAL, OUTPUT,
                                     PACKAGE, ROOT, candidate_count, row_count,
                                     run, sha256, verify_extra)

EXPECTED_NAME_EXTRA = 237_217
BACKUP = OUTPUT / "final_us_address_verified"


def main():
    marker = OUTPUT / "US_ADDRESS_REVISION_COMPLETE.json"
    if not marker.is_file():
        raise ValueError("Verify the completed US-address submission first")
    base_candidate = OUTPUT / "generalized_us_address_candidate_pairs.tsv"
    base_raw = OUTPUT / "generalized_us_address_results_uncapped.tsv"
    gate = ROOT / "tmp/us_name_gate2_ids.txt"
    extra = OUTPUT / "us_name_tfidf_top10_gated.tsv"
    candidate = OUTPUT / "generalized_us_both_candidate_pairs.tsv"
    raw = OUTPUT / "generalized_us_both_results_uncapped.tsv"
    capped = OUTPUT / "generalized_us_both_results_capped.tsv"
    matching = OUTPUT / "generalized_us_both_results.tsv"
    doc = ROOT / "docs/Documentation_generalized_us_both.md"
    archive = OUTPUT / "Vulcans_us_both_submission.zip"
    pdf = OUTPUT / "pdf/Vulcans_approach_summary_both.pdf"
    if any(row_count(path) != EXPECTED for path in (base_candidate, base_raw)):
        raise ValueError("Verified US-address base inputs are incomplete")
    verify_extra(extra, gate, expected=EXPECTED_NAME_EXTRA, max_k=10)
    run(PACKAGE / "src/merge_address_candidates.py", "--data-dir", DATA,
        "--db", DB, "--base-candidate", base_candidate,
        "--base-matching", base_raw, "--extra", extra,
        "--candidate", candidate, "--matching", raw,
        "--model", PACKAGE / "generalized_model.joblib",
        "--blank-specialist", PACKAGE / "blank_frequency_model.joblib",
        "--country", "US", "--general-threshold", "0.75",
        "--blank-threshold", "0.8", "--sparse-extra", "--workers", "4")
    if row_count(candidate) != EXPECTED or row_count(raw) != EXPECTED:
        raise ValueError("US-name candidate/result rows are incomplete")
    run(PACKAGE / "src/cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "generalized_model.joblib", "--input", raw,
        "--output", capped, "--cap", "11")
    run(PACKAGE / "src/resolve_exclusivity.py", "--data-dir", DATA,
        "--db", DB, "--model", PACKAGE / "generalized_model.joblib",
        "--input", capped, "--output", matching)
    count = candidate_count(candidate)
    content = doc.read_text(encoding="utf-8")
    pattern = r"(\*\*Final test last-stage candidate pairs:\*\* )[^\n]+?( across 1,732,544 test Source 1 records)"
    revised, substitutions = re.subn(pattern, rf"\g<1>{count:,}\g<2>", content)
    if substitutions != 1:
        raise ValueError("Cannot update methodology candidate count")
    doc.write_text(revised, encoding="utf-8")
    if not pdf.is_file():
        raise ValueError("Revised two-page approach PDF is missing")
    run(ROOT / "analysis/package_submission.py", "--team-name", "Vulcans",
        "--check-ids", "--variant", "compact_generalized",
        "--matching-path", matching, "--candidate-path", candidate,
        "--doc-path", doc, "--archive-path", archive,
        "--compresslevel", "9")
    size = archive.stat().st_size
    if size >= 512_000_000:
        raise ValueError(f"Final archive exceeds 512 MB portal cap: {size}")
    if not FINAL.is_dir():
        raise ValueError("Expected verified US-address final directory")
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
    result = {"queries": EXPECTED, "gated_name_queries": EXPECTED_NAME_EXTRA,
              "last_stage_candidate_pairs": count, "archive_bytes": size,
              "matching_sha256": sha256(matching),
              "candidate_sha256": sha256(candidate),
              "archive_sha256": sha256(archive),
              "final_dir": str(FINAL), "fallback_dir": str(BACKUP)}
    (OUTPUT / "US_BOTH_REVISION_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
