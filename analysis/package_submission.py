"""Validate and create the exact final challenge archive."""

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT / "code" / "business_entity_resolution"
OUTPUT=ROOT / "output"
RESOURCE=ROOT / "6ab10eb3b23ba_student_resource" / "student_resource"
sys.path.insert(0,str(PACKAGE / "src"))
from stream_validate import verify  # noqa: E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--team-name",required=True)
    parser.add_argument("--check-ids",action="store_true")
    args=parser.parse_args()
    safe_name=re.sub(r"[^A-Za-z0-9_-]+","_",args.team_name.strip()).strip("_")
    if not safe_name:
        raise SystemExit("Provide the registered team name")
    doc=ROOT / "Documentation_template.md"
    content=doc.read_text(encoding="utf-8")
    if "[Registered team name]" in content or "[Registered member names]" in content or "[Fill after complete run" in content:
        raise SystemExit("Fill the methodology template before packaging")
    matching=OUTPUT / "matching_results.tsv"
    candidates=OUTPUT / "candidate_pairs.tsv"
    verify(RESOURCE / "dataset" / "test",matching,candidates)
    # The supplied validator retains every candidate ID in Python sets and
    # exceeds this machine's RAM at full scale. The streaming check above
    # covers both TSVs; run the supplied validator on matches and IDs.
    command=[sys.executable,str(RESOURCE / "utils" / "validate_submission.py"),
             "--matching",str(matching),"--candidate",str(OUTPUT / "__no_candidates__.tsv"),
             "--test-dir",str(RESOURCE / "dataset" / "test")]
    if args.check_ids:
        command.append("--check-ids")
    subprocess.run(command,check=True)
    source_files=[p for p in (PACKAGE / "src").glob("*.py")]
    required=[PACKAGE / "README.md",PACKAGE / "requirements.txt",PACKAGE / "model.joblib",doc,matching,candidates]
    for path in source_files+required:
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    archive=OUTPUT / f"{safe_name}_submission.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as bundle:
        for path in (matching,candidates):
            bundle.write(path,path.relative_to(ROOT).as_posix())
        for path in [PACKAGE / "README.md",PACKAGE / "requirements.txt",PACKAGE / "model.joblib"]+source_files:
            bundle.write(path,path.relative_to(ROOT).as_posix())
        bundle.write(doc,doc.name)
    with zipfile.ZipFile(archive) as bundle:
        bad=bundle.testzip()
        if bad:
            raise SystemExit(f"Corrupt archive member: {bad}")
        print("archive",archive,"bytes",archive.stat().st_size,"members",len(bundle.namelist()))


if __name__=="__main__":
    main()
