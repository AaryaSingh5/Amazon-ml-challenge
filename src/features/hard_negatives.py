"""Stage 12: Hard Negative Mining & Refinement.

Finds false positive predictions (high-confidence non-matches) from the current
model and adds them as informative training negatives to improve precision.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple

from src.utils.storage import StorageManager
from src.training.train import load_dataset, extract_matrix

logger = logging.getLogger(__name__)


def run_hard_negatives(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Mines informative non-matching candidate pairs to improve model precision."""
    logger.info("Starting Stage 12: Hard Negative Mining...")
    storage.initialize_directories()

    try:
        import joblib
    except ImportError:
        logger.error("joblib not installed.")
        return {"status": "error", "message": "joblib missing"}

    # 1. Load Model
    model_path = storage.checkpoints_dir / "latest" / "model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run Stage 9 first.")
        return {}

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # 2. Load features
    feat_tsv = storage.features_dir / "train_features.tsv"
    if not feat_tsv.exists():
        logger.error(f"Features file missing at {feat_tsv}")
        return {}

    train_rows, val_rows, feature_names = load_dataset(feat_tsv)
    all_rows = train_rows + val_rows
    logger.info(f"Total candidate pairs: {len(all_rows):,} (train: {len(train_rows):,}, val: {len(val_rows):,})")

    if not all_rows:
        logger.warning("No data found.")
        return {}

    X_all, y_all = extract_matrix(all_rows, feature_names)

    # 3. Get Model Predictions
    if hasattr(model, "best_iteration"):
        preds = model.predict(X_all, num_iteration=model.best_iteration)
    elif hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_all)[:, 1]
    else:
        preds = model.predict(X_all)

    # 4. Load Optimal Threshold
    opt_th_file = storage.artifacts_dir / "optimal_thresholds.json"
    threshold = 0.5
    if opt_th_file.exists():
        with open(opt_th_file, "r", encoding="utf-8") as f:
            opt_data = json.load(f)
            threshold = opt_data.get("optimal_threshold", 0.5)
    logger.info(f"Using decision threshold: {threshold}")

    # 5. Load Ground Truth to identify FP/FN
    gt_file = storage.raw_dir / config["dataset"]["files"].get("train_ground_truth", "train_ground_truth.tsv")
    gt_map: Dict[str, Set[str]] = defaultdict(set)

    if gt_file.exists():
        with open(gt_file, "r", encoding=config["dataset"].get("encoding", "utf-8"), errors="replace") as f:
            reader = csv.reader(f, delimiter=config["dataset"].get("separator", "\t"))
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                s1 = row[0].strip()
                targets = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 else []
                gt_map[s1].update(targets)

    # 6. Identify Hard Negatives: False Positives predicted above a high confidence
    #    Hard Negatives are non-match pairs where model still gives a high confidence score
    #    These are confusing/ambiguous cases — very valuable for re-training
    hard_neg_threshold = max(threshold, 0.3)  # Pairs predicted positive but are actually negative
    
    hard_neg_rows: List[Dict[str, Any]] = []
    stats = {
        "total_pairs": len(all_rows),
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "hard_negatives_mined": 0,
    }

    for row, pred, true_label in zip(all_rows, preds, y_all):
        pred_label = 1 if pred >= threshold else 0

        if true_label == 1 and pred_label == 1:
            stats["true_positives"] += 1
        elif true_label == 0 and pred_label == 1:
            stats["false_positives"] += 1
            # This is a hard negative: model is confused about this non-match pair
            if pred >= hard_neg_threshold:
                hard_neg_row = dict(row)
                hard_neg_row["is_hard_negative"] = 1
                hard_neg_row["model_score"] = round(float(pred), 4)
                hard_neg_rows.append(hard_neg_row)
        elif true_label == 1 and pred_label == 0:
            stats["false_negatives"] += 1
        else:
            stats["true_negatives"] += 1

    stats["hard_negatives_mined"] = len(hard_neg_rows)
    logger.info(f"Mined {len(hard_neg_rows):,} hard negative pairs (model confusions).")

    # 7. Create Augmented Feature Dataset: original features + hard negatives (marked)
    #    Hard negatives are relabeled to is_match=0 and assigned extra weight
    #    Read original rows, then append hard negatives with a repeated weight flag
    out_tsv = storage.features_dir / "train_features_mined.tsv"
    
    with open(feat_tsv, "r", encoding="utf-8", errors="replace") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        original_fieldnames = reader.fieldnames or []
        original_rows = list(reader)

    mined_fieldnames = original_fieldnames + ["is_hard_negative", "model_score"]

    with open(out_tsv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=mined_fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()

        # Write all original rows with default hard negative flags
        for row in original_rows:
            row["is_hard_negative"] = 0
            row["model_score"] = ""
            writer.writerow(row)

        # Append hard negatives (with model_score annotations)
        writer.writerows(hard_neg_rows)

    logger.info(f"Augmented dataset saved to {out_tsv}")

    # 8. Save Report
    report = {
        "threshold_used": threshold,
        "hard_neg_score_cutoff": hard_neg_threshold,
        "stats": stats,
        "output_path": str(out_tsv),
        "total_augmented_pairs": len(original_rows) + len(hard_neg_rows),
    }

    out_json = storage.features_dir / "hard_negatives_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Hard negative mining report saved to {out_json}")
    return report
