"""Build and validate the Unicode-aware portal-size-limited submission.

This runs from the already saved broad test candidates. It never uploads files.
Restarting after interruption skips completed full-length intermediate TSVs.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource"
DATA = RESOURCE / "dataset" / "test"
DB = ROOT / ".duckdb" / "test_index.duckdb"
PACKAGE = ROOT / "code" / "business_entity_resolution"
OUTPUT = ROOT / "output"
FINAL = OUTPUT / "final"
EXPECTED = 1_732_544
TOP_K = 40


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def row_count(path):
    if not path.is_file():
        return -1
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count - 1


def candidate_count(path):
    count = 0
    with path.open("rb") as stream:
        next(stream)
        for row in stream:
            values = row.partition(b"\t")[2].strip()
            if values:
                count += values.count(b",") + 1
    return count


def main():
    OUTPUT.mkdir(exist_ok=True)
    FINAL.mkdir(exist_ok=True)
    broad = OUTPUT / "candidate_pairs.tsv"
    compact = OUTPUT / "generalized_compact_candidate_pairs.tsv"
    raw = OUTPUT / "generalized_compact_results_uncapped.tsv"
    capped = OUTPUT / "generalized_compact_results_capped.tsv"
    final = OUTPUT / "generalized_compact_results.tsv"
    marker = OUTPUT / "GENERALIZED_COMPLETE.json"
    archive = OUTPUT / "Vulcans_generalized_compact_submission.zip"
    if marker.is_file():
        print("Compact submission already complete:", marker, flush=True)
        return
    if row_count(broad) != EXPECTED:
        raise ValueError("Broad candidate file is incomplete")
    if row_count(compact) != EXPECTED or row_count(raw) != EXPECTED:
        run(PACKAGE / "src" / "two_stage_rescore.py", "--data-dir", DATA,
            "--db", DB, "--broad-candidate", broad, "--candidate", compact,
            "--matching", raw, "--top-k", str(TOP_K), "--workers", "4",
            "--matcher-model", PACKAGE / "generalized_model.joblib",
            "--blank-specialist-model", PACKAGE / "blank_frequency_model.joblib",
            "--blank-threshold", "0.8",
            "--threshold", "0.75", "--country-threshold", "India:0.65")
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "generalized_model.joblib", "--input", raw,
        "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "generalized_model.joblib", "--input", capped,
        "--output", final)
    run(PACKAGE / "src" / "stream_validate.py", "--test-dir", DATA,
        "--matching", final, "--candidate", compact)
    count = candidate_count(compact)
    doc = ROOT / "docs" / "Documentation_compact_generalized.md"
    content = doc.read_text(encoding="utf-8")
    placeholder = "[Fill after compact run and validator]"
    replacement = f"{count:,}"
    if placeholder in content:
        doc.write_text(content.replace(placeholder, replacement), encoding="utf-8")
    elif f"**Final test last-stage candidate pairs:** {replacement}" not in content:
        raise ValueError("Unexpected compact candidate count in methodology")
    run(ROOT / "analysis" / "package_submission.py", "--team-name", "Vulcans",
        "--check-ids", "--variant", "compact_generalized")
    archive_size = archive.stat().st_size
    if archive_size >= 512_000_000:
        raise ValueError(f"Compact archive exceeds 512 MB portal limit: {archive_size}")
    shutil.copyfile(archive, FINAL / "Vulcans_submission.zip")
    shutil.copyfile(final, FINAL / "matching_results.tsv")
    pdf = OUTPUT / "pdf" / "Vulcans_approach_summary.pdf"
    if pdf.is_file():
        shutil.copyfile(pdf, FINAL / pdf.name)
    result = {"queries": EXPECTED, "last_stage_candidate_pairs": count,
              "archive": str(FINAL / "Vulcans_submission.zip"),
              "archive_bytes": archive_size,
              "leaderboard_tsv": str(FINAL / "matching_results.tsv"),
              "approach_pdf": str(FINAL / pdf.name)}
    marker.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
