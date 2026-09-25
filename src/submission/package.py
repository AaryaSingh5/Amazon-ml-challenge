"""Stage 15: Submission Verification and Packaging."""
from __future__ import annotations

import logging
from typing import Dict, Any
from src.utils.storage import StorageManager
from src.submission.validator import validate_submission_files

logger = logging.getLogger(__name__)


def run_submission_packaging(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Validates final submission files and formats them according to competition guidelines."""
    logger.info("Starting Stage 15: Submission Packaging & Verification...")
    matching_tsv = storage.output_dir / "matching_results.tsv"
    candidate_tsv = storage.output_dir / "candidate_pairs.tsv"
    test_s1 = storage.raw_dir / "test_source1.tsv"

    is_valid, issues = validate_submission_files(
        matching_tsv_path=matching_tsv,
        candidate_tsv_path=candidate_tsv,
        test_dir_or_s1_path=test_s1,
    )
    return {"valid": is_valid, "issues": issues}
