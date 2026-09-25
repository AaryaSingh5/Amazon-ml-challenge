"""Unit tests for Macro F0.5, Precision, Recall, and Singleton handling."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_entity_fbeta, compute_macro_f05, compute_candidate_recall


def test_true_singleton_correct():
    # True singleton predicted as empty -> Full credit (1.0)
    res = compute_entity_fbeta(set(), set(), beta=0.5)
    assert res["f_beta"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_true_singleton_false_positive():
    # True singleton predicted with false matches -> Penalized (0.0)
    res = compute_entity_fbeta(set(), {"S2_100"}, beta=0.5)
    assert res["f_beta"] == 0.0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_non_singleton_empty_prediction():
    # True match exists, but predicted empty -> Penalized (0.0)
    res = compute_entity_fbeta({"S2_100"}, set(), beta=0.5)
    assert res["f_beta"] == 0.0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_f05_precision_weighting():
    # 1 TP, 1 FP (Precision = 0.5, Recall = 1.0)
    # F0.5 = 1.25 * (0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
    res = compute_entity_fbeta({"S2_100"}, {"S2_100", "S3_200"}, beta=0.5)
    assert abs(res["precision"] - 0.5) < 1e-5
    assert abs(res["recall"] - 1.0) < 1e-5
    assert abs(res["f_beta"] - (0.625 / 1.125)) < 1e-5


def test_macro_f05_average():
    gt = {
        "S1_1": {"S2_10"},       # Case 1: perfect match (F=1.0)
        "S1_2": set(),            # Case 2: true singleton, pred empty (F=1.0)
        "S1_3": set(),            # Case 3: true singleton, pred false match (F=0.0)
        "S1_4": {"S2_30"},       # Case 4: non-singleton, pred empty (F=0.0)
    }
    preds = {
        "S1_1": {"S2_10"},
        "S1_2": set(),
        "S1_3": {"S3_99"},
        "S1_4": set(),
    }
    res = compute_macro_f05(gt, preds, beta=0.5)
    assert res["total_entities"] == 4
    assert res["total_singletons"] == 2
    assert res["correct_singletons"] == 1
    assert abs(res["singleton_accuracy"] - 0.5) < 1e-5
    # (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5
    assert abs(res["macro_f0.5"] - 0.50) < 1e-5


if __name__ == "__main__":
    test_true_singleton_correct()
    test_true_singleton_false_positive()
    test_non_singleton_empty_prediction()
    test_f05_precision_weighting()
    test_macro_f05_average()
    print("All metric unit tests passed successfully!")
