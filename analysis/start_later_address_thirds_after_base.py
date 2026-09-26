"""Launch two exact address-search shards after base scoring releases RAM."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PACKAGE = ROOT / "code" / "business_entity_resolution"
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
EXPECTED = 1_732_544


def row_count(path):
    if not path.is_file():
        return -1
    with path.open("rb") as stream:
        return sum(chunk.count(b"\n") for chunk in
                   iter(lambda: stream.read(4 * 1024 * 1024), b"")) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pid", type=int, required=True)
    args = parser.parse_args()
    while psutil.pid_exists(args.base_pid):
        time.sleep(15)
    # Wait for the scorer's four feature workers to release their large arrays.
    time.sleep(30)
    base_candidate = OUTPUT / "generalized_base_candidate_pairs.tsv"
    base_raw = OUTPUT / "generalized_base_results_uncapped.tsv"
    if row_count(base_candidate) != EXPECTED or row_count(base_raw) != EXPECTED:
        raise RuntimeError("Base scorer ended without complete output; shards not launched")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    pids = []
    for number in (2, 3):
        shard_output = OUTPUT / f"india_address_tfidf_shard_{number}.tsv"
        if shard_output.exists():
            raise RuntimeError(f"Shard {number} output already exists")
        command = [sys.executable,
                   str(PACKAGE / "src" / "address_tfidf_retrieval.py"),
                   "--data-dir", str(DATA), "--output", str(shard_output),
                   "--top-k", "20", "--query-ids",
                   str(ROOT / "tmp" / f"india_address_shard_{number}_ids.txt")]
        with (OUTPUT / f"india_address_tfidf_shard_{number}.log").open("w") as stdout, \
             (OUTPUT / f"india_address_tfidf_shard_{number}.err.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout,
                                       stderr=stderr, creationflags=flags)
        (ROOT / "tmp" / f"india_address_shard_{number}_pid.txt").write_text(
            str(process.pid) + "\n", encoding="utf-8")
        pids.append(process.pid)
        print("STARTED_SHARD", number, process.pid, flush=True)
    print("STARTED_LATER_THIRDS", *pids, flush=True)


if __name__ == "__main__":
    main()
