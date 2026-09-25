"""Stage 10: Entity Decision and Threshold Optimization."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Set, List

from src.utils.storage import StorageManager
from src.evaluation.metrics import compute_macro_f05
from src.training.train import load_dataset, extract_matrix

logger = logging.getLogger(__name__)


def run_threshold_optimization(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Optimizes decision thresholds to maximize macro F0.5 on validation split."""
    logger.info("Starting Stage 10: Threshold Optimization...")
    storage.initialize_directories()

    try:
        import joblib
    except ImportError:
        logger.error("joblib not installed. Cannot load ML model.")
        return {"status": "error", "message": "joblib missing"}

    # 1. Load Model
    model_path = storage.checkpoints_dir / "latest" / "model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run Stage 9 first.")
        return {}

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # 2. Load Validation Features
    feat_tsv = storage.features_dir / "train_features.tsv"
    if not feat_tsv.exists():
        logger.error(f"Features file missing at {feat_tsv}")
        return {}

    _, val_rows, feature_names = load_dataset(feat_tsv)
    if not val_rows:
        logger.warning("No validation data found in features.")
        return {}

    X_val, _ = extract_matrix(val_rows, feature_names)

    # 3. Predict Probabilities
    logger.info(f"Predicting probabilities for {len(val_rows)} validation pairs...")
    
    # Check if LightGBM model (it has best_iteration) or scikit-learn model
    if hasattr(model, "best_iteration"):
        preds = model.predict(X_val, num_iteration=model.best_iteration)
    elif hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_val)[:, 1]
    else:
        preds = model.predict(X_val)

    # 4. Load Ground Truth
    val_ids: Set[str] = set()
    split_file = storage.splits_dir / "validation_split.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            val_ids = set(json.load(f).get("val_entity_ids", []))

    gt_file = storage.raw_dir / config["dataset"]["files"].get("train_ground_truth", "train_ground_truth.tsv")
    gt_map: Dict[str, Set[str]] = {s1: set() for s1 in val_ids}

    if gt_file.exists():
        with open(gt_file, "r", encoding=config["dataset"].get("encoding", "utf-8"), errors="replace") as f:
            reader = csv.reader(f, delimiter=config["dataset"].get("separator", "\t"))
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                s1 = row[0].strip()
                if s1 in val_ids:
                    targets = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 else []
                    gt_map[s1].update(targets)

    # 5. Grid Search Thresholds
    grid_cfg = config.get("evaluation", {}).get("threshold_grid_search", {})
    start_th = grid_cfg.get("start", 0.20)
    end_th = grid_cfg.get("end", 0.95)
    step_th = grid_cfg.get("step", 0.02)
    beta = config.get("pipeline", {}).get("beta", 0.5)

    num_steps = int(round((end_th - start_th) / step_th))
    thresholds = [round(start_th + i * step_th, 4) for i in range(num_steps + 1)]

    logger.info(f"Grid searching {len(thresholds)} thresholds from {start_th} to {end_th} (step {step_th})...")

    results = []
    best_f_beta = -1.0
    best_threshold = 0.5
    best_metrics = {}

    for th in thresholds:
        # Generate predictions for this threshold
        val_preds: Dict[str, Set[str]] = defaultdict(set)
        for row, pred in zip(val_rows, preds):
            if pred >= th:
                val_preds[row["s1_entity_id"]].add(row["target_entity_id"])

        # Evaluate
        metrics = compute_macro_f05(ground_truth=gt_map, predictions=val_preds, beta=beta)
        f_beta = metrics.get(f"macro_f{beta}", 0.0)

        results.append({
            "threshold": th,
            "macro_f_beta": f_beta,
            "macro_precision": metrics.get("macro_precision", 0.0),
            "macro_recall": metrics.get("macro_recall", 0.0),
            "singleton_accuracy": metrics.get("singleton_accuracy", 0.0)
        })

        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_threshold = th
            best_metrics = metrics

    logger.info(f"Optimal Threshold: {best_threshold} (F{beta}: {best_f_beta:.4f})")
    
    # 6. Save Results
    report = {
        "optimal_threshold": best_threshold,
        "best_macro_f_beta": best_f_beta,
        "best_metrics": best_metrics,
        "grid_search_results": results
    }

    out_json = storage.artifacts_dir / "optimal_thresholds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Threshold optimization report saved to {out_json}")
    
    return report
