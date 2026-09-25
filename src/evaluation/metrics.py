"""Evaluation metrics for Business Entity Resolution.

Implements Macro-Averaged F0.5 over Source 1 entities, precision, recall,
singleton scoring, candidate recall, and error breakdown.
"""

from __future__ import annotations

from typing import Dict, Set, List, Any, Union


def compute_entity_fbeta(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Computes Precision, Recall, and F-beta score for a single Source 1 entity.

    Rules:
    - True Singleton (true_matches is empty):
        - If pred_matches is also empty -> TP=0, FP=0, FN=0 -> Precision=1.0, Recall=1.0, F_beta=1.0.
        - If pred_matches is not empty  -> FP > 0           -> Precision=0.0, Recall=0.0, F_beta=0.0.
    - Non-Singleton (true_matches is not empty):
        - If pred_matches is empty      -> FN > 0           -> Precision=0.0, Recall=0.0, F_beta=0.0.
        - Otherwise compute standard precision, recall, and F_beta.
    """
    beta_sq = beta ** 2

    # Case 1: True Singleton
    if len(true_matches) == 0:
        if len(pred_matches) == 0:
            return {"precision": 1.0, "recall": 1.0, "f_beta": 1.0, "tp": 0, "fp": 0, "fn": 0}
        else:
            return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0, "tp": 0, "fp": len(pred_matches), "fn": 0}

    # Case 2: Non-Singleton, Empty Prediction
    if len(pred_matches) == 0:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0, "tp": 0, "fp": 0, "fn": len(true_matches)}

    # Case 3: Non-Singleton, Non-Empty Prediction
    tp = len(true_matches.intersection(pred_matches))
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (beta_sq * precision) + recall
    if denom > 0.0:
        f_beta = (1.0 + beta_sq) * (precision * recall) / denom
    else:
        f_beta = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, Any]:
    """Computes Macro-Averaged F0.5 across all Source 1 entities in ground truth.

    Args:
        ground_truth: Mapping from s1_entity_id -> set of matching s2/s3 entity_ids.
        predictions: Mapping from s1_entity_id -> set of predicted matching s2/s3 entity_ids.
        beta: F-score weight (default 0.5).

    Returns:
        Dictionary with macro_f0.5, macro_precision, macro_recall, singleton_accuracy, etc.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f_beta": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "total_entities": 0}

    sum_p = 0.0
    sum_r = 0.0
    sum_f = 0.0

    total_singletons = 0
    correct_singletons = 0
    total_non_singletons = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores = compute_entity_fbeta(true_set, pred_set, beta=beta)

        sum_p += scores["precision"]
        sum_r += scores["recall"]
        sum_f += scores["f_beta"]
        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]

        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
        else:
            total_non_singletons += 1

    macro_f = sum_f / total_entities
    macro_p = sum_p / total_entities
    macro_r = sum_r / total_entities
    singleton_acc = (correct_singletons / total_singletons) if total_singletons > 0 else 1.0

    return {
        f"macro_f{beta}": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "total_entities": total_entities,
        "total_singletons": total_singletons,
        "correct_singletons": correct_singletons,
        "singleton_accuracy": singleton_acc,
        "total_non_singletons": total_non_singletons,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def compute_candidate_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Computes Candidate Recall: true pairs captured by candidate generation vs total true pairs."""
    total_true_pairs = 0
    captured_true_pairs = 0

    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        cand_set = candidates.get(s1_id, set())
        total_true_pairs += len(true_set)
        captured_true_pairs += len(true_set.intersection(cand_set))

    candidate_recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0

    return {
        "candidate_recall": candidate_recall,
        "captured_true_pairs": captured_true_pairs,
        "total_true_pairs": total_true_pairs,
        "missed_true_pairs": total_true_pairs - captured_true_pairs,
    }
