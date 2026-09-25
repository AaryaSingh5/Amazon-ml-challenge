"""Unit tests for Stage 5: Candidate Generation and Blocking module."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.rules import BlockingEngine
from src.preprocessing.normalizer import transform_record


def test_blocking_engine_multi_strategy_and_no_s1_s1():
    # 1. Create clean records
    s1_raw = [
        transform_record("S1_1", "Apple Inc", "1 Infinite Loop", "US"),
        transform_record("S1_2", "Cafe Coffee Day", "MG Road 560001", "India"),
        transform_record("S1_3", "Random Isolated Business", "Some Unknown Rd", "US"),
    ]

    s2_raw = [
        # S2_1: matches S1_1 via clean name ("apple")
        transform_record("S2_1", "Apple LLC", "1 Infinite Loop Ste 100", "US"),
        # S2_2: matches S1_2 via rare token + address postal ("560001")
        transform_record("S2_2", "Coffee Day Express", "MG Road 560001", "India"),
    ]

    s3_raw = [
        # S3_1: matches S1_1 via exact address
        transform_record("S3_1", "Apple Computer", "1 Infinite Loop", "US"),
    ]

    engine = BlockingEngine(max_candidates_per_entity=10)
    engine.fit_targets(s2_records=s2_raw, s3_records=s3_raw)

    # Test S1_1 candidates
    cands_s1_1 = engine.generate_candidates_for_s1(s1_raw[0])
    target_ids_s1_1 = {t_id for t_id, _ in cands_s1_1}

    # Should capture both S2_1 and S3_1
    assert "S2_1" in target_ids_s1_1
    assert "S3_1" in target_ids_s1_1
    # Must NEVER contain any S1 ID
    for t_id in target_ids_s1_1:
        assert not t_id.startswith("S1_")

    # Test S1_2 candidates
    cands_s1_2 = engine.generate_candidates_for_s1(s1_raw[1])
    target_ids_s1_2 = {t_id for t_id, _ in cands_s1_2}
    assert "S2_2" in target_ids_s1_2

    # Verify triggered rules are recorded
    rules_dict = {t_id: rules for t_id, rules in cands_s1_1}
    assert "rule_clean_name" in rules_dict["S2_1"] or "rule_prefix_tokens" in rules_dict["S2_1"]


def test_blocking_deduplication():
    s1_rec = transform_record("S1_1", "Walmart Supercenter", "702 SW 8th St", "US")
    s2_rec = transform_record("S2_100", "Walmart Supercenter", "702 Southwest 8th Street", "US")

    engine = BlockingEngine()
    engine.fit_targets(s2_records=[s2_rec], s3_records=[])

    cands = engine.generate_candidates_for_s1(s1_rec)
    target_ids = [t_id for t_id, _ in cands]
    # Ensure S2_100 appears exactly ONCE even if matched by multiple rules
    assert target_ids.count("S2_100") == 1


if __name__ == "__main__":
    test_blocking_engine_multi_strategy_and_no_s1_s1()
    test_blocking_deduplication()
    print("All Stage 5 Blocking unit tests passed successfully!")
