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
    parser.add_argument("--variant",choices=["baseline","number","compact_number","compact_generalized"],default="baseline")
    args=parser.parse_args()
    safe_name=re.sub(r"[^A-Za-z0-9_-]+","_",args.team_name.strip()).strip("_")
    if not safe_name:
        raise SystemExit("Provide the registered team name")
    is_generalized=args.variant=="compact_generalized"
    is_number=args.variant in ("number","compact_number")
    is_compact=args.variant in ("compact_number","compact_generalized")
    doc=(ROOT / "docs" / "Documentation_compact_generalized.md" if is_generalized else
         ROOT / "docs" / "Documentation_compact_number.md" if is_compact else
         ROOT / "docs" / "Documentation_number_model.md" if is_number else
         ROOT / "Documentation_template.md")
    if not doc.is_file():
        raise SystemExit(f"Missing methodology document: {doc}")
    content=doc.read_text(encoding="utf-8")
    if ("[Registered team name]" in content or "[Registered member names]" in content or
            "[Fill after complete run" in content or "[Fill after compact run" in content):
        raise SystemExit("Fill the methodology template before packaging")
    matching=OUTPUT / ("generalized_compact_results.tsv" if is_generalized else
                       "compact_number_results.tsv" if is_compact else
                       "number_results.tsv" if is_number else "matching_results.tsv")
    candidates=OUTPUT / ("generalized_compact_candidate_pairs.tsv" if is_generalized else
                         "compact_candidate_pairs.tsv" if is_compact else "candidate_pairs.tsv")
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
    models=[PACKAGE / "model.joblib"]
    if is_number:
        models.append(PACKAGE / "number_model.joblib")
    if is_generalized:
        models.append(PACKAGE / "generalized_model.joblib")
        models.append(PACKAGE / "generalized_seed_model.joblib")
        models.append(PACKAGE / "blank_frequency_model.joblib")
    required=[PACKAGE / "README.md",PACKAGE / "requirements.txt",*models,doc,matching,candidates]
    for path in source_files+required:
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    suffix=("_generalized_compact_submission.zip" if is_generalized else
            "_compact_submission.zip" if is_compact else
            "_number_submission.zip" if is_number else "_submission.zip")
    archive=OUTPUT / f"{safe_name}{suffix}"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as bundle:
        bundle.write(matching,"output/matching_results.tsv")
        bundle.write(candidates,"output/candidate_pairs.tsv")
        for path in [PACKAGE / "README.md",PACKAGE / "requirements.txt",*models]+source_files:
            bundle.write(path,path.relative_to(ROOT).as_posix())
        bundle.write(doc,"Documentation_template.md")
    with zipfile.ZipFile(archive) as bundle:
        bad=bundle.testzip()
        if bad:
            raise SystemExit(f"Corrupt archive member: {bad}")
        print("archive",archive,"bytes",archive.stat().st_size,"members",len(bundle.namelist()))


if __name__=="__main__":
    main()
