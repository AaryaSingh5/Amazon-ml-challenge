"""Official Submission Validator for Amazon ML Entity Resolution Challenge.

Enforces all schema, entity presence, subset constraint, and formatting rules.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple

logger = logging.getLogger(__name__)


def validate_submission_files(
    matching_tsv_path: str | Path,
    candidate_tsv_path: str | Path,
    test_dir_or_s1_path: str | Path,
    test_s2_path: str | Path | None = None,
    test_s3_path: str | Path | None = None,
) -> Tuple[bool, List[str]]:
    """Validates matching_results.tsv and candidate_pairs.tsv against test entities.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []
    m_path = Path(matching_tsv_path)
    c_path = Path(candidate_tsv_path)

    if not m_path.exists():
        return False, [f"Matching file does not exist: {m_path}"]
    if not c_path.exists():
        return False, [f"Candidate file does not exist: {c_path}"]

    # 1. Read Test Source 1 IDs
    s1_expected_ids: Set[str] = set()
    s2_valid_ids: Set[str] = set()
    s3_valid_ids: Set[str] = set()

    test_input = Path(test_dir_or_s1_path)
    if test_input.is_dir():
        s1_file = test_input / "test_source1.tsv"
        s2_file = test_input / "test_source2.tsv"
        s3_file = test_input / "test_source3.tsv"
    else:
        s1_file = test_input
        s2_file = Path(test_s2_path) if test_s2_path else None
        s3_file = Path(test_s3_path) if test_s3_path else None

    if s1_file.exists():
        with open(s1_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if row:
                    s1_expected_ids.add(row[0].strip())

    if s2_file and s2_file.exists():
        with open(s2_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    s2_valid_ids.add(row[0].strip())

    if s3_file and s3_file.exists():
        with open(s3_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    s3_valid_ids.add(row[0].strip())

    valid_match_pool = s2_valid_ids.union(s3_valid_ids)

    # 2. Validate Candidate Pairs TSV
    candidate_map: Dict[str, Set[str]] = {}
    seen_cand_s1: Set[str] = set()

    with open(c_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            issues.append(f"Candidate file header invalid or not tab-delimited: {header}")
        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            cand_str = row[1].strip() if len(row) > 1 else ""

            if s1_id in seen_cand_s1:
                issues.append(f"Duplicate S1 ID in candidate file at line {line_num}: {s1_id}")
            seen_cand_s1.add(s1_id)

            cand_list = [x.strip() for x in cand_str.split(",") if x.strip()]
            if len(cand_list) != len(set(cand_list)):
                issues.append(f"Duplicate candidate IDs for entity {s1_id} at line {line_num}")

            candidate_map[s1_id] = set(cand_list)

    # 3. Validate Matching Results TSV
    matching_map: Dict[str, Set[str]] = {}
    seen_match_s1: Set[str] = set()

    with open(m_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            issues.append(f"Matching file header invalid or not tab-delimited: {header}")
        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            match_str = row[1].strip() if len(row) > 1 else ""

            if s1_id in seen_match_s1:
                issues.append(f"Duplicate S1 ID in matching file at line {line_num}: {s1_id}")
            seen_match_s1.add(s1_id)

            match_list = [x.strip() for x in match_str.split(",") if x.strip()]
            if len(match_list) != len(set(match_list)):
                issues.append(f"Duplicate match IDs for entity {s1_id} at line {line_num}")

            # Verify matches are subset of candidates
            cand_set = candidate_map.get(s1_id, set())
            for m_id in match_list:
                if m_id not in cand_set:
                    issues.append(f"Match ID '{m_id}' for entity '{s1_id}' is NOT present in candidate pairs!")
                if valid_match_pool and m_id not in valid_match_pool:
                    issues.append(f"Match ID '{m_id}' does not exist in test Source 2 or Source 3!")
                if s1_expected_ids and m_id in s1_expected_ids:
                    issues.append(f"Source 1 ID '{m_id}' incorrectly listed as a matched target!")

            matching_map[s1_id] = set(match_list)

    # 4. Check coverage of Source 1 entities
    if s1_expected_ids:
        missing_in_match = s1_expected_ids - seen_match_s1
        if missing_in_match:
            issues.append(f"Missing {len(missing_in_match)} Source 1 test entities in matching file!")
        extra_in_match = seen_match_s1 - s1_expected_ids
        if extra_in_match:
            issues.append(f"Found {len(extra_in_match)} unexpected entities in matching file!")

    is_valid = len(issues) == 0
    return is_valid, issues
