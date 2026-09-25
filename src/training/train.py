"""Stage 9: First Machine Learning Matching Model."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple


from src.utils.storage import StorageManager
from src.evaluation.metrics import compute_macro_f05

logger = logging.getLogger(__name__)


def load_dataset(features_file: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Loads features TSV and splits into train and validation rows."""
    train_rows = []
    val_rows = []
    features_list = []
    
    with open(features_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return [], [], []
            
        exclude_cols = {"s1_entity_id", "target_entity_id", "is_match", "is_validation"}
        features_list = [c for c in reader.fieldnames if c not in exclude_cols]
        
        for row in reader:
            if row.get("is_validation") == "1":
                val_rows.append(row)
            else:
                train_rows.append(row)
                
    return train_rows, val_rows, features_list


def extract_matrix(rows: List[Dict[str, Any]], feature_names: List[str]) -> Tuple[list, list]:
    """Converts a list of dict rows to a 2D feature matrix and label vector."""
    X = []
    y = []
    for row in rows:
        row_feats = []
        for fn in feature_names:
            try:
                val = float(row[fn])
            except (ValueError, TypeError):
                val = 0.0
            row_feats.append(val)
        X.append(row_feats)
        
        try:
            label = int(row.get("is_match", 0))
        except (ValueError, TypeError):
            label = 0
        y.append(label)
        
    return X, y


def run_training(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Trains primary machine learning model on candidate pair features."""
    logger.info("Starting Stage 9: First ML Matching Model...")
    storage.initialize_directories()
    
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Cannot train ML model.")
        return {"status": "error", "message": "LightGBM not installed"}
    
    # 1. Load data
    feat_tsv = storage.features_dir / "train_features.tsv"
    if not feat_tsv.exists():
        logger.error(f"Features file missing at {feat_tsv}")
        return {}
        
    logger.info(f"Loading features from {feat_tsv}...")
    train_rows, val_rows, feature_names = load_dataset(feat_tsv)
    
    if not train_rows:
        logger.warning("No training data found. Make sure 'is_validation' is correctly populated.")
        return {}
        
    logger.info(f"Train samples: {len(train_rows):,}, Validation samples: {len(val_rows):,}")
    logger.info(f"Using {len(feature_names)} features.")
    
    X_train, y_train = extract_matrix(train_rows, feature_names)
    X_val, y_val = extract_matrix(val_rows, feature_names)
    
    # 2. Train Model
    gbm_params = config.get("models", {}).get("gbm", {})
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": gbm_params.get("learning_rate", 0.05),
        "num_leaves": gbm_params.get("num_leaves", 31),
        "max_depth": gbm_params.get("max_depth", -1),
        "subsample": gbm_params.get("subsample", 0.8),
        "colsample_bytree": gbm_params.get("colsample_bytree", 0.8),
        "scale_pos_weight": gbm_params.get("scale_pos_weight", 1.0),
        "random_state": gbm_params.get("random_state", 42),
        "verbose": -1,
    }
    
    n_estimators = gbm_params.get("n_estimators", 500)
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    logger.info("Training LightGBM model...")
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=gbm_params.get("early_stopping_rounds", 30), verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    
    # 3. Predict on Validation
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Fixed threshold for Stage 9 (Stage 10 will optimize)
    threshold = 0.5
    
    val_predictions: Dict[str, Set[str]] = defaultdict(set)
    for row, pred in zip(val_rows, preds):
        if pred >= threshold:
            val_predictions[row["s1_entity_id"]].add(row["target_entity_id"])
            
    # 4. Evaluate using Validation Ground Truth
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
                    
    beta = config["pipeline"].get("beta", 0.5)
    metrics = compute_macro_f05(ground_truth=gt_map, predictions=val_predictions, beta=beta)
    logger.info(f"First ML Model F{beta} Score (Threshold {threshold}): {metrics.get(f'macro_f{beta}', 0):.4f}")
    
    # Feature Importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    
    # 5. Save Artifacts
    exp_dir = storage.experiments_dir / "exp_001_first_ml"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "model_type": "lightgbm",
        "best_iteration": model.best_iteration,
        "evaluation_threshold": threshold,
        "metrics": metrics,
        "feature_importance_gain": [
            {"feature": f, "gain": float(g)} for f, g in feat_imp[:15]
        ]
    }
    
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    checkpoints_dir = storage.checkpoints_dir / "latest"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = checkpoints_dir / "model.joblib"
    
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return report
