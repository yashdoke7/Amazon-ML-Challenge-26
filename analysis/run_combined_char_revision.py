"""Build and verify the US-address + India-name character retrieval revision.

The currently verified output/final is preserved until all checks pass.
"""

import csv
import json
import re
import shutil

from run_us_address_revision import (DATA, DB, EXPECTED, FINAL, OUTPUT,
                                     PACKAGE, ROOT, candidate_count, row_count,
                                     run, sha256)


BACKUP = OUTPUT / "final_us_both_verified"
COUNTS = {"US": 663_106, "India": 809_986}


def verify_extra(path, country, max_k):
    expected = COUNTS[country]
    if row_count(path) != expected:
        raise ValueError(f"Incomplete {country} retrieval: {path}")
    with (DATA / "test_source1.tsv").open(encoding="utf-8", newline="") as source, \
         path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"Unexpected retrieval header: {path}")
        count = 0
        for query in csv.DictReader(source, delimiter="\t"):
            if query["country"] != country:
                continue
            row = next(reader, None)
            if row is None or row["source1_entity_id"] != query["entity_id"]:
                raise ValueError(f"{country} retrieval row/order mismatch at {count}")
            ids = row["candidate_entity_ids"].split(",") if row["candidate_entity_ids"] else []
            if len(ids) > max_k or len(ids) != len(set(ids)):
                raise ValueError(f"Invalid {country} candidate quota at {query['entity_id']}")
            count += 1
        if count != expected or next(reader, None) is not None:
            raise ValueError(f"{country} retrieval count mismatch")
    print("verified retrieval", country, count, flush=True)


def merge(base_candidate, base_raw, extra, candidate, raw, country, threshold):
    run(PACKAGE / "src/merge_address_candidates.py", "--data-dir", DATA,
        "--db", DB, "--base-candidate", base_candidate,
        "--base-matching", base_raw, "--extra", extra,
        "--candidate", candidate, "--matching", raw,
        "--model", PACKAGE / "generalized_model.joblib",
        "--blank-specialist", PACKAGE / "blank_frequency_model.joblib",
        "--country", country, "--general-threshold", str(threshold),
        "--blank-threshold", "0.8", "--sparse-extra", "--workers", "4")
    if row_count(candidate) != EXPECTED or row_count(raw) != EXPECTED:
        raise ValueError(f"Incomplete merged output after {country}")


def main():
    if not (OUTPUT / "US_BOTH_REVISION_COMPLETE.json").is_file():
        raise ValueError("Expected verified US address-plus-name base")
    base_candidate = OUTPUT / "generalized_us_both_candidate_pairs.tsv"
    base_raw = OUTPUT / "generalized_us_both_results_uncapped.tsv"
    us_extra = OUTPUT / "us_char_address_top10.tsv"
    india_extra = OUTPUT / "india_char_name_top5.tsv"
    us_candidate = OUTPUT / "generalized_us_char_candidate_pairs.tsv"
    us_raw = OUTPUT / "generalized_us_char_results_uncapped.tsv"
    candidate = OUTPUT / "generalized_combined_char_candidate_pairs.tsv"
    raw = OUTPUT / "generalized_combined_char_results_uncapped.tsv"
    capped = OUTPUT / "generalized_combined_char_results_capped.tsv"
    matching = OUTPUT / "generalized_combined_char_results.tsv"
    doc = ROOT / "docs/Documentation_generalized_combined_char.md"
    pdf = OUTPUT / "pdf/Vulcans_approach_summary_combined_char.pdf"
    archive = OUTPUT / "Vulcans_combined_char_submission.zip"
    if any(row_count(path) != EXPECTED for path in (base_candidate, base_raw)):
        raise ValueError("Verified base candidate/raw files are incomplete")
    verify_extra(us_extra, "US", 10)
    verify_extra(india_extra, "India", 5)
    merge(base_candidate, base_raw, us_extra, us_candidate, us_raw, "US", 0.75)
    merge(us_candidate, us_raw, india_extra, candidate, raw, "India", 0.65)
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
        raise ValueError("Updated approach PDF missing")
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
    result = {"queries": EXPECTED, "us_char_queries": COUNTS["US"],
              "india_name_queries": COUNTS["India"],
              "last_stage_candidate_pairs": count, "archive_bytes": size,
              "matching_sha256": sha256(matching),
              "candidate_sha256": sha256(candidate),
              "archive_sha256": sha256(archive),
              "final_dir": str(FINAL), "fallback_dir": str(BACKUP)}
    (OUTPUT / "COMBINED_CHAR_REVISION_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
