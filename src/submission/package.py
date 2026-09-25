"""Stage 15: Submission Verification and Packaging.

Validates matching_results.tsv and candidate_pairs.tsv against the test source
entities, then bundles everything into a submission_package.zip.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.utils.storage import StorageManager
from src.submission.validator import validate_submission_files

logger = logging.getLogger(__name__)


def run_submission_packaging(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Validates final submission files and packages them into a zip."""
    logger.info("Starting Stage 15: Submission Packaging & Verification...")
    storage.initialize_directories()

    matching_tsv = storage.output_dir / "matching_results.tsv"
    candidate_tsv = storage.output_dir / "candidate_pairs.tsv"
    test_s1 = storage.raw_dir / config["dataset"]["files"].get("test_source1", "test_source1.tsv")
    test_s2 = storage.raw_dir / config["dataset"]["files"].get("test_source2", "test_source2.tsv")
    test_s3 = storage.raw_dir / config["dataset"]["files"].get("test_source3", "test_source3.tsv")

    # 1. Validate submission files
    logger.info("Validating submission files...")
    is_valid, issues = validate_submission_files(
        matching_tsv_path=matching_tsv,
        candidate_tsv_path=candidate_tsv,
        test_dir_or_s1_path=test_s1,
        test_s2_path=test_s2 if test_s2.exists() else None,
        test_s3_path=test_s3 if test_s3.exists() else None,
    )

    if issues:
        logger.warning(f"Submission validation found {len(issues)} issue(s):")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Submission validation PASSED — no issues found.")

    # 2. Compute quick stats on matching_results.tsv
    num_entities = 0
    num_matched = 0
    num_singletons = 0
    if matching_tsv.exists():
        with open(matching_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                num_entities += 1
                match_ids = [x for x in (row[1].split(",") if len(row) > 1 else []) if x.strip()]
                if match_ids:
                    num_matched += 1
                else:
                    num_singletons += 1

    validation_report = {
        "is_valid": is_valid,
        "issues": issues,
        "num_entities": num_entities,
        "num_entities_with_matches": num_matched,
        "num_singletons": num_singletons,
        "validated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Save validation report
    val_report_path = storage.artifacts_dir / "submission_validation_report.json"
    val_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(f"Validation report saved to {val_report_path}")

    # 3. Print summary
    print("\n" + "=" * 60)
    print("  SUBMISSION VALIDATION REPORT")
    print("=" * 60)
    print(f"  Status         : {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"  Total Entities : {num_entities:,}")
    print(f"  Matched        : {num_matched:,}")
    print(f"  Singletons     : {num_singletons:,}")
    if issues:
        print(f"  Issues ({len(issues)}):")
        for issue in issues[:5]:
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues)-5} more.")
    print("=" * 60 + "\n")

    return {"valid": is_valid, "issues": issues, "stats": validation_report}
