"""Stage 11: Advanced Modeling Experiments.

Trains multiple gradient boosting models (XGBoost, CatBoost, stacked ensemble)
on the hard-negative-mined feature set and saves the best model as checkpoints/best/model.joblib.
Falls back gracefully if any library is not installed.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple, Optional

from src.utils.storage import StorageManager
from src.evaluation.metrics import compute_macro_f05
from src.training.train import extract_matrix

logger = logging.getLogger(__name__)


def _load_mined_or_fallback(storage: StorageManager) -> Tuple[str, List[Dict], List[Dict], List[str]]:
    """Load hard-negative-mined dataset if available, otherwise fall back to base features."""
    mined_tsv = storage.features_dir / "train_features_mined.tsv"
    base_tsv = storage.features_dir / "train_features.tsv"

    target = mined_tsv if mined_tsv.exists() else base_tsv
    label = "mined" if mined_tsv.exists() else "base"

    train_rows, val_rows = [], []
    feature_names = []

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return label, [], [], []
        exclude = {"s1_entity_id", "target_entity_id", "is_match", "is_validation", "is_hard_negative", "model_score"}
        feature_names = [c for c in reader.fieldnames if c not in exclude]
        for row in reader:
            if row.get("is_validation") == "1":
                val_rows.append(row)
            else:
                train_rows.append(row)

    logger.info(f"Loaded '{label}' dataset: {len(train_rows):,} train, {len(val_rows):,} val, {len(feature_names)} features.")
    return label, train_rows, val_rows, feature_names


def _evaluate_predictions(
    predictions: Dict[str, Set[str]],
    gt_map: Dict[str, Set[str]],
    beta: float,
) -> Dict[str, Any]:
    return compute_macro_f05(ground_truth=gt_map, predictions=predictions, beta=beta)


def _predict_at_threshold(val_rows, preds, threshold: float) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = defaultdict(set)
    for row, pred in zip(val_rows, preds):
        if pred >= threshold:
            result[row["s1_entity_id"]].add(row["target_entity_id"])
    return result


def _load_gt_and_val_ids(config, storage) -> Tuple[Set[str], Dict[str, Set[str]]]:
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
    return val_ids, gt_map


def run_advanced_training(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Trains advanced ensembles and calibrated models."""
    logger.info("Starting Stage 11: Advanced Model Experiments...")
    storage.initialize_directories()

    # Load optimal threshold from Stage 10
    opt_th_file = storage.artifacts_dir / "optimal_thresholds.json"
    threshold = 0.5
    if opt_th_file.exists():
        with open(opt_th_file, "r") as f:
            threshold = json.load(f).get("optimal_threshold", 0.5)
    logger.info(f"Decision threshold: {threshold}")

    # Load data
    data_label, train_rows, val_rows, feature_names = _load_mined_or_fallback(storage)
    if not train_rows:
        logger.error("No training data found.")
        return {"status": "error", "message": "No training data"}

    X_train, y_train = extract_matrix(train_rows, feature_names)
    X_val, y_val = extract_matrix(val_rows, feature_names)

    beta = config.get("pipeline", {}).get("beta", 0.5)
    val_ids, gt_map = _load_gt_and_val_ids(config, storage)

    experiment_results = []
    best_f_beta = -1.0
    best_model = None
    best_model_name = ""

    # -----------------------------------------------------------------------
    # Experiment A: LightGBM re-trained on mined dataset
    # -----------------------------------------------------------------------
    try:
        import lightgbm as lgb
        gbm_params = config.get("models", {}).get("gbm", {})
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": gbm_params.get("learning_rate", 0.05),
            "num_leaves": gbm_params.get("num_leaves", 63),  # Increased for stage 11
            "max_depth": gbm_params.get("max_depth", -1),
            "subsample": gbm_params.get("subsample", 0.8),
            "colsample_bytree": gbm_params.get("colsample_bytree", 0.8),
            "min_child_samples": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
        lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=700,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)]
        )
        lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        lgb_predictions = _predict_at_threshold(val_rows, lgb_preds, threshold)
        lgb_metrics = _evaluate_predictions(lgb_predictions, gt_map, beta)
        f = lgb_metrics.get(f"macro_f{beta}", 0.0)
        logger.info(f"[LightGBM-v2 ({data_label})] F{beta}={f:.4f}")
        experiment_results.append({"model": f"lightgbm_{data_label}", "macro_f_beta": f, "metrics": lgb_metrics})
        if f > best_f_beta:
            best_f_beta, best_model, best_model_name = f, lgb_model, f"lightgbm_{data_label}"
    except ImportError:
        logger.warning("LightGBM not installed; skipping LightGBM experiment.")
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")

    # -----------------------------------------------------------------------
    # Experiment B: XGBoost
    # -----------------------------------------------------------------------
    try:
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        xgb_params = {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "seed": 42, "verbosity": 0,
        }
        xgb_model = xgb.train(
            xgb_params, dtrain, num_boost_round=700,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=40, verbose_eval=False
        )
        xgb_preds = xgb_model.predict(dval)
        xgb_predictions = _predict_at_threshold(val_rows, xgb_preds, threshold)
        xgb_metrics = _evaluate_predictions(xgb_predictions, gt_map, beta)
        f = xgb_metrics.get(f"macro_f{beta}", 0.0)
        logger.info(f"[XGBoost] F{beta}={f:.4f}")
        experiment_results.append({"model": "xgboost", "macro_f_beta": f, "metrics": xgb_metrics})
        if f > best_f_beta:
            best_f_beta, best_model, best_model_name = f, xgb_model, "xgboost"
    except ImportError:
        logger.warning("XGBoost not installed; skipping XGBoost experiment.")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")

    # -----------------------------------------------------------------------
    # Experiment C: CatBoost
    # -----------------------------------------------------------------------
    try:
        from catboost import CatBoostClassifier
        cb_model = CatBoostClassifier(
            iterations=700, learning_rate=0.05, depth=6,
            eval_metric="Logloss", random_seed=42, verbose=False,
            early_stopping_rounds=40
        )
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        cb_preds = cb_model.predict_proba(X_val)[:, 1]
        cb_predictions = _predict_at_threshold(val_rows, cb_preds, threshold)
        cb_metrics = _evaluate_predictions(cb_predictions, gt_map, beta)
        f = cb_metrics.get(f"macro_f{beta}", 0.0)
        logger.info(f"[CatBoost] F{beta}={f:.4f}")
        experiment_results.append({"model": "catboost", "macro_f_beta": f, "metrics": cb_metrics})
        if f > best_f_beta:
            best_f_beta, best_model, best_model_name = f, cb_model, "catboost"
    except ImportError:
        logger.warning("CatBoost not installed; skipping CatBoost experiment.")
    except Exception as e:
        logger.error(f"CatBoost training failed: {e}")

    if best_model is None:
        logger.error("No models trained successfully.")
        return {"status": "error", "message": "No models trained"}

    logger.info(f"Best model: {best_model_name} (F{beta}={best_f_beta:.4f})")

    # Save best model
    best_dir = storage.checkpoints_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_model, best_dir / "model.joblib")
    logger.info(f"Best model saved to {best_dir / 'model.joblib'}")

    # Save experiment report
    exp_dir = storage.experiments_dir / "exp_advanced"
    exp_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "best_model_name": best_model_name,
        "best_macro_f_beta": best_f_beta,
        "data_used": data_label,
        "experiments": experiment_results,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(storage.checkpoints_dir / "best" / "model_meta.json", "w") as f:
        json.dump({"model_name": best_model_name, "macro_f_beta": best_f_beta, "threshold": threshold}, f, indent=2)

    logger.info(f"Advanced modeling report saved to {exp_dir / 'metrics.json'}")
    return report
