"""Finish both local challenge archives once the active baseline inference is complete.

Safe to start during inference. This never uploads to the competition portal and
keeps the raw baseline matches in matching_results_uncapped.tsv.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource"
DATA = RESOURCE / "dataset" / "test"
DB = ROOT / ".duckdb" / "test_index.duckdb"
PACKAGE = ROOT / "code" / "business_entity_resolution"
OUTPUT = ROOT / "output"
MATCHING = OUTPUT / "matching_results.tsv"
CANDIDATES = OUTPUT / "candidate_pairs.tsv"


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def line_count(path):
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def candidate_count(path):
    count = 0
    with path.open("rb") as stream:
        next(stream)
        for row in stream:
            values = row.partition(b"\t")[2].strip()
            if values:
                count += values.count(b",") + 1
    return count


def wait_for_inference(timeout_hours):
    source_path = DATA / "test_source1.tsv"
    expected = line_count(source_path) - 1
    print("expected Source 1 rows", expected, flush=True)
    started = time.monotonic()
    previous_sizes = None
    while time.monotonic() - started < timeout_hours * 3600:
        if MATCHING.is_file() and CANDIDATES.is_file():
            sizes = (MATCHING.stat().st_size, CANDIDATES.stat().st_size)
            if sizes == previous_sizes and line_count(MATCHING) == expected + 1:
                if line_count(CANDIDATES) == expected + 1:
                    print("baseline inference appears complete", sizes, flush=True)
                    return expected
            previous_sizes = sizes
        time.sleep(60)
    raise TimeoutError("Baseline inference did not complete within the wait period")


def fill_methodology(total_candidates, total_queries):
    replacement = f"{total_candidates:,} across {total_queries:,} test Source 1 records"
    for path in (ROOT / "Documentation_template.md",
                 ROOT / "docs" / "Documentation_number_model.md"):
        content = path.read_text(encoding="utf-8")
        placeholder = "[Fill after complete run and validator]"
        if placeholder in content:
            path.write_text(content.replace(placeholder, replacement), encoding="utf-8")
        elif replacement not in content:
            raise ValueError(f"Methodology has an unexpected candidate count: {path}")
    print("filled methodology candidate count", replacement, flush=True)


def make_baseline():
    raw = OUTPUT / "matching_results_uncapped.tsv"
    archive = OUTPUT / "Vulcans_submission.zip"
    marker = OUTPUT / "BASELINE_COMPLETE.json"
    if raw.is_file() and archive.is_file() and marker.is_file():
        print("baseline archive already exists", archive, flush=True)
        return
    if not raw.is_file():
        shutil.copyfile(MATCHING, raw)
    capped = OUTPUT / "matching_results_capped.tsv"
    owned = OUTPUT / "matching_results_owned.tsv"
    final_temp = OUTPUT / "matching_results_final.tmp.tsv"
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "model.joblib", "--input", raw,
        "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "model.joblib", "--input", capped,
        "--output", owned)
    run(PACKAGE / "src" / "veto_nearby_number.py", "--data-dir", DATA, "--db", DB,
        "--input", owned, "--output", final_temp)
    os.replace(final_temp, MATCHING)
    run(ROOT / "analysis" / "package_submission.py", "--team-name", "Vulcans",
        "--check-ids")
    marker.write_text(json.dumps({"archive": str(archive),
                                  "bytes": archive.stat().st_size}) + "\n", encoding="utf-8")
    print("baseline archive ready", archive, flush=True)


def make_number(expected):
    raw = OUTPUT / "number_results_uncapped.tsv"
    if not raw.is_file() or line_count(raw) != expected + 1:
        run(PACKAGE / "src" / "rescore_candidates.py", "--data-dir", DATA,
            "--db", DB, "--model", PACKAGE / "number_model.joblib",
            "--candidate", CANDIDATES, "--output", raw, "--threshold", "0.75",
            "--country-threshold", "India:0.65", "--workers", "4")
    capped = OUTPUT / "number_results_capped.tsv"
    final = OUTPUT / "number_results.tsv"
    run(PACKAGE / "src" / "cap_predictions.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "number_model.joblib", "--input", raw,
        "--output", capped, "--cap", "11")
    run(PACKAGE / "src" / "resolve_exclusivity.py", "--data-dir", DATA, "--db", DB,
        "--model", PACKAGE / "number_model.joblib", "--input", capped,
        "--output", final)
    run(ROOT / "analysis" / "package_submission.py", "--team-name", "Vulcans",
        "--check-ids", "--variant", "number")
    comparison = OUTPUT / "country_output_comparison.json"
    with comparison.open("w", encoding="utf-8") as stream:
        subprocess.run([
            sys.executable, str(ROOT / "analysis" / "compare_unlabeled_output_tails.py"),
            "--source1", str(DATA / "test_source1.tsv"),
            "--reference", str(MATCHING), "--alternative", str(final),
        ], cwd=ROOT, check=True, stdout=stream)
    print("number-model archive ready", OUTPUT / "Vulcans_number_submission.zip", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-hours", type=float, default=12)
    args = parser.parse_args()
    expected = wait_for_inference(args.wait_hours)
    run(PACKAGE / "src" / "stream_validate.py", "--test-dir", DATA,
        "--matching", MATCHING, "--candidate", CANDIDATES)
    total_candidates = candidate_count(CANDIDATES)
    fill_methodology(total_candidates, expected)
    make_baseline()
    make_number(expected)
    marker = OUTPUT / "FINALIZATION_COMPLETE.json"
    marker.write_text(json.dumps({
        "queries": expected,
        "candidate_pairs": total_candidates,
        "baseline_zip_bytes": (OUTPUT / "Vulcans_submission.zip").stat().st_size,
        "number_zip_bytes": (OUTPUT / "Vulcans_number_submission.zip").stat().st_size,
        "comparison": str(OUTPUT / "country_output_comparison.json"),
    }, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", marker, flush=True)


if __name__ == "__main__":
    main()
