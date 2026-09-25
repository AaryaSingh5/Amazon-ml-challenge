"""Unit tests for Stage 6: Candidate Recall Evaluation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.candidate_recall import evaluate_candidate_recall_details


def test_candidate_recall_ablation_and_sources():
    gt_map = {
        "S1_1": {"S2_10", "S3_20"},  # 2 matches
        "S1_2": {"S2_30"},           # 1 match
        "S1_3": set(),               # Singleton
        "S1_4": {"S3_40"},           # 1 match (missed)
    }

    cand_pairs = [
        {"s1_entity_id": "S1_1", "target_entity_id": "S2_10", "rules_triggered": "rule_exact_name,rule_clean_name"},
        {"s1_entity_id": "S1_1", "target_entity_id": "S3_20", "rules_triggered": "rule_exact_addr"},
        {"s1_entity_id": "S1_2", "target_entity_id": "S2_30", "rules_triggered": "rule_clean_name"},
        {"s1_entity_id": "S1_2", "target_entity_id": "S2_99", "rules_triggered": "rule_char_3gram"}, # Non-matching candidate
    ]

    val_entities = {"S1_1", "S1_3"}

    res = evaluate_candidate_recall_details(
        ground_truth_map=gt_map,
        candidate_pairs_list=cand_pairs,
        val_entity_ids=val_entities,
    )

    assert res["total_true_pairs"] == 4  # S2_10, S3_20, S2_30, S3_40
    assert res["captured_true_pairs"] == 3
    assert res["missed_true_pairs"] == 1
    assert res["overall_candidate_recall"] == 75.0

    # Validation recall: S1_1 has 2 matches, both captured -> 100%
    assert res["val_candidate_recall"] == 100.0

    # Source-level recall
    assert res["source2_recall"] == 100.0  # 2/2 captured
    assert res["source3_recall"] == 50.0   # 1/2 captured

    # Rule ablation
    ablation = res["rule_ablation_analysis"]
    assert "rule_exact_addr" in ablation
    # S3_20 was captured ONLY by rule_exact_addr -> unique contribution = 1
    assert ablation["rule_exact_addr"]["unique_contribution_pairs"] == 1


if __name__ == "__main__":
    test_candidate_recall_ablation_and_sources()
    print("All Stage 6 Candidate Recall unit tests passed successfully!")
