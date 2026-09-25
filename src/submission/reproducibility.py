"""Stage 16: Final Reproducibility Package Builder.

Creates a self-contained submission_package.zip containing:
  - output/matching_results.tsv        (primary submission file)
  - output/candidate_pairs.tsv         (required companion)
  - artifacts/submission_validation_report.json
  - artifacts/final_model_selection.json
  - artifacts/optimal_thresholds.json
  - requirements.txt
  - configs/config.yaml
  - README.md
  - git commit hash (reproducibility_info.json)
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


def _get_git_info() -> Dict[str, str]:
    """Try to get git commit hash and branch."""
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    return info


def run_reproducibility_package(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Builds the final submission zip with submission files, artifacts and metadata."""
    logger.info("Starting Stage 16: Final Reproducibility Packaging...")
    storage.initialize_directories()

    git_info = _get_git_info()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pkg_name = f"submission_package_{timestamp}.zip"
    pkg_path = Path(pkg_name)

    # Files to include: (source_path, archive_name)
    files_to_bundle = [
        (storage.output_dir / "matching_results.tsv",                "output/matching_results.tsv"),
        (storage.output_dir / "candidate_pairs.tsv",                 "output/candidate_pairs.tsv"),
        (storage.artifacts_dir / "submission_validation_report.json","artifacts/submission_validation_report.json"),
        (storage.artifacts_dir / "final_model_selection.json",       "artifacts/final_model_selection.json"),
        (storage.artifacts_dir / "optimal_thresholds.json",          "artifacts/optimal_thresholds.json"),
        (Path("requirements.txt"),                                    "requirements.txt"),
        (Path("configs/config.yaml"),                                 "configs/config.yaml"),
        (Path("README.md"),                                           "README.md"),
    ]

    reproducibility_meta = {
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "packaged_at": datetime.utcnow().isoformat() + "Z",
        "pipeline": config.get("pipeline", {}),
    }

    included = []
    missing = []

    with zipfile.ZipFile(pkg_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write reproducibility info JSON
        zf.writestr("reproducibility_info.json", json.dumps(reproducibility_meta, indent=2))

        for src, arc_name in files_to_bundle:
            src = Path(src)
            if src.exists():
                zf.write(src, arc_name)
                included.append(arc_name)
                logger.info(f"  + Added: {arc_name}")
            else:
                missing.append(arc_name)
                logger.warning(f"  - Missing (skipped): {arc_name}")

    pkg_size_kb = pkg_path.stat().st_size / 1024

    print("\n" + "=" * 60)
    print("  REPRODUCIBILITY PACKAGE COMPLETE")
    print("=" * 60)
    print(f"  Package : {pkg_path}")
    print(f"  Size    : {pkg_size_kb:.1f} KB")
    print(f"  Commit  : {git_info['commit'][:12]}")
    print(f"  Branch  : {git_info['branch']}")
    print(f"  Included: {len(included)} files")
    if missing:
        print(f"  Missing : {missing}")
    print("=" * 60 + "\n")

    report = {
        "package_path": str(pkg_path),
        "package_size_kb": round(pkg_size_kb, 1),
        "files_included": included,
        "files_missing": missing,
        "git_info": git_info,
        "packaged_at": reproducibility_meta["packaged_at"],
    }

    return report
