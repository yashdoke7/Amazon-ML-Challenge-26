"""Finish, validate, and package the measured gated US-address revision.

Preserves the verified India-only submission until the new ZIP passes every
contract check. Never uploads anything to the portal.
"""

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "6ab10eb3b23ba_student_resource/student_resource"
DATA = RESOURCE / "dataset/test"
DB = ROOT / ".duckdb/test_index.duckdb"
PACKAGE = ROOT / "code/business_entity_resolution"
OUTPUT = ROOT / "output"
FINAL = OUTPUT / "final"
BACKUP = OUTPUT / "final_india_only_0p926444"
EXPECTED = 1_732_544
EXPECTED_US_EXTRA = 399_606


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def row_count(path):
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count - 1


def candidate_count(path):
    count = 0
    with path.open("rb") as stream:
        next(stream)
        for line in stream:
            values = line.partition(b"\t")[2].strip()
            if values:
                count += values.count(b",") + 1
    return count


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_extra(extra, gate):
    if row_count(extra) != EXPECTED_US_EXTRA:
        raise ValueError("US retrieval output is partial; do not merge")
    with gate.open(encoding="utf-8") as gate_stream, \
         extra.open(encoding="utf-8", newline="") as extra_stream:
        reader = csv.DictReader(extra_stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Unexpected US retrieval header")
        count = 0
        for wanted, row in zip(gate_stream, reader):
            q = wanted.strip()
            if q != row["source1_entity_id"]:
                raise ValueError(f"US retrieval order mismatch at {count}")
            ids = row["candidate_entity_ids"].split(",") if row[
                "candidate_entity_ids"] else []
            if len(ids) > 5 or len(ids) != len(set(ids)):
                raise ValueError(f"Invalid US candidate quota at {q}")
            count += 1
        if count != EXPECTED_US_EXTRA or next(gate_stream, None) is not None or \
                next(reader, None) is not None:
            raise ValueError("US retrieval/gate counts differ")
    print("verified US retrieval rows", count, flush=True)


def main():
    base_candidate = OUTPUT / "generalized_compact_candidate_pairs.tsv"
    base_raw = OUTPUT / "generalized_compact_results_uncapped.tsv"
    gate = ROOT / "tmp/us_address_gate_ids.txt"
    extra = OUTPUT / "us_address_tfidf_top5_gated.tsv"
    candidate = OUTPUT / "generalized_us_address_candidate_pairs.tsv"
    raw = OUTPUT / "generalized_us_address_results_uncapped.tsv"
    capped = OUTPUT / "generalized_us_address_results_capped.tsv"
    matching = OUTPUT / "generalized_us_address_results.tsv"
    doc = ROOT / "docs/Documentation_generalized_us_address.md"
    archive = OUTPUT / "Vulcans_us_address_submission.zip"
    pdf = OUTPUT / "pdf/Vulcans_approach_summary.pdf"
    if any(row_count(path) != EXPECTED for path in (base_candidate, base_raw)):
        raise ValueError("Verified India-only base inputs are incomplete")
    verify_extra(extra, gate)
    run(PACKAGE / "src/merge_address_candidates.py", "--data-dir", DATA,
        "--db", DB, "--base-candidate", base_candidate,
        "--base-matching", base_raw, "--extra", extra,
        "--candidate", candidate, "--matching", raw,
        "--model", PACKAGE / "generalized_model.joblib",
        "--blank-specialist", PACKAGE / "blank_frequency_model.joblib",
        "--country", "US", "--general-threshold", "0.75",
        "--blank-threshold", "0.8", "--sparse-extra", "--workers", "4")
    if row_count(candidate) != EXPECTED or row_count(raw) != EXPECTED:
        raise ValueError("Merged US candidate/result rows incomplete")
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
        raise ValueError("Revised approach PDF is missing")
    run(ROOT / "analysis/package_submission.py", "--team-name", "Vulcans",
        "--check-ids", "--variant", "compact_generalized",
        "--matching-path", matching, "--candidate-path", candidate,
        "--doc-path", doc, "--archive-path", archive)
    size = archive.stat().st_size
    if size >= 512_000_000:
        raise ValueError(f"Archive exceeds 512 MB portal cap: {size}")
    if not FINAL.is_dir():
        raise ValueError("Expected verified India-only fallback directory")
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
    result = {"queries": EXPECTED, "gated_us_queries": EXPECTED_US_EXTRA,
              "last_stage_candidate_pairs": count, "archive_bytes": size,
              "matching_sha256": sha256(matching),
              "candidate_sha256": sha256(candidate),
              "archive_sha256": sha256(archive),
              "final_dir": str(FINAL), "fallback_dir": str(BACKUP)}
    (OUTPUT / "US_ADDRESS_REVISION_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
