"""Stage 14: Test Inference Pipeline.

End-to-end inference on test set:
  1. Loads best model + optimal threshold from Stage 13 artifacts
  2. Runs blocking (candidate generation) on the test sources
  3. Extracts pairwise features for all test candidate pairs
  4. Predicts match probability and thresholds to binary match/no-match
  5. Writes output/matching_results.tsv and output/candidate_pairs.tsv
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Set, List

from src.utils.storage import StorageManager
from src.blocking.pipeline import run_blocking
from src.features.pipeline import run_features
from src.training.train import load_dataset, extract_matrix

logger = logging.getLogger(__name__)


def _load_best_model_and_threshold(storage: StorageManager):
    """Load the best model checkpoint and optimal threshold."""
    try:
        import joblib
    except ImportError:
        raise RuntimeError("joblib not installed.")

    # Prefer best checkpoint, fall back to latest
    selection_json = storage.artifacts_dir / "final_model_selection.json"
    optimal_threshold = 0.5

    if selection_json.exists():
        with open(selection_json, "r") as f:
            sel = json.load(f)
        model_path = Path(sel.get("best_checkpoint_path", ""))
        optimal_threshold = sel.get("optimal_threshold", 0.5)
    else:
        model_path = storage.checkpoints_dir / "best" / "model.joblib"
        if not model_path.exists():
            model_path = storage.checkpoints_dir / "latest" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"No model checkpoint found. Train a model first (Stages 9-11).")

    logger.info(f"Loading model from: {model_path}  (threshold={optimal_threshold})")
    model = joblib.load(model_path)
    return model, optimal_threshold


def run_inference(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Runs complete end-to-end inference on test set and writes TSVs."""
    logger.info("Starting Stage 14: Test Inference...")
    storage.initialize_directories()

    # 1. Load model + threshold
    try:
        model, threshold = _load_best_model_and_threshold(storage)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        return {"status": "error", "message": str(e)}

    # 2. Generate test candidates (Stage 5 re-run in test mode)
    logger.info("Generating test candidate pairs via blocking...")
    run_blocking(config, storage, is_test=True)

    # 3. Extract test features (Stage 7 re-run in test mode)
    logger.info("Extracting features for test candidate pairs...")
    run_features(config, storage, is_test=True)

    # 4. Load test features
    test_feat_tsv = storage.features_dir / "test_features.tsv"
    if not test_feat_tsv.exists():
        logger.error(f"Test features not found at {test_feat_tsv}")
        return {"status": "error", "message": "test_features.tsv missing"}

    # Parse test features
    test_rows: List[Dict[str, Any]] = []
    feature_names: List[str] = []
    with open(test_feat_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames:
            exclude = {"s1_entity_id", "target_entity_id"}
            feature_names = [c for c in reader.fieldnames if c not in exclude]
        for row in reader:
            test_rows.append(row)

    if not test_rows:
        logger.warning("No test candidate pairs found. Outputting empty submission.")
        # Write empty but valid outputs
        _write_empty_submission(config, storage)
        return {"status": "empty_submission", "num_pairs": 0}

    logger.info(f"Loaded {len(test_rows):,} test candidate pairs with {len(feature_names)} features.")

    # Build feature matrix
    X_test, _ = extract_matrix(test_rows, feature_names)

    # 5. Predict
    if hasattr(model, "best_iteration"):
        preds = model.predict(X_test, num_iteration=model.best_iteration)
    elif hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "predict"):
        # XGBoost DMatrix-based model — need DMatrix wrapper
        try:
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test, feature_names=feature_names)
            preds = model.predict(dtest)
        except Exception:
            preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)

    # 6. Threshold → binary decisions → aggregate per S1 entity
    candidate_pairs: Dict[str, Set[str]] = defaultdict(set)
    matching_results: Dict[str, Set[str]] = defaultdict(set)

    for row, pred in zip(test_rows, preds):
        s1 = row["s1_entity_id"]
        t = row["target_entity_id"]
        candidate_pairs[s1].add(t)
        if float(pred) >= threshold:
            matching_results[s1].add(t)

    # 7. Load all test S1 entity IDs to ensure full coverage (singletons need empty predictions)
    all_s1_ids: Set[str] = set()
    sep = config["dataset"].get("separator", "\t")
    enc = config["dataset"].get("encoding", "utf-8")
    s1_test_file = storage.raw_dir / config["dataset"]["files"].get("test_source1", "test_source1.tsv")
    if s1_test_file.exists():
        with open(s1_test_file, "r", encoding=enc, errors="replace") as f:
            reader = csv.reader(f, delimiter=sep)
            next(reader, None)
            for row in reader:
                if row:
                    all_s1_ids.add(row[0].strip())
    else:
        all_s1_ids = set(candidate_pairs.keys())

    logger.info(f"Test S1 entities: {len(all_s1_ids):,} | Predicted matches for: {len(matching_results):,}")

    # 8. Write output files
    storage.output_dir.mkdir(parents=True, exist_ok=True)

    matching_tsv = storage.output_dir / "matching_results.tsv"
    candidate_tsv = storage.output_dir / "candidate_pairs.tsv"

    with open(matching_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["s1_entity_id", "matching_ids"])
        for s1 in sorted(all_s1_ids):
            matches = matching_results.get(s1, set())
            writer.writerow([s1, ",".join(sorted(matches))])

    with open(candidate_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["s1_entity_id", "candidate_ids"])
        for s1 in sorted(all_s1_ids):
            cands = candidate_pairs.get(s1, set())
            writer.writerow([s1, ",".join(sorted(cands))])

    num_matched = sum(1 for v in matching_results.values() if v)
    num_singletons = len(all_s1_ids) - len(matching_results)

    report = {
        "status": "success",
        "num_test_entities": len(all_s1_ids),
        "num_test_candidate_pairs": len(test_rows),
        "num_entities_with_matches": num_matched,
        "num_predicted_singletons": num_singletons,
        "threshold_used": threshold,
        "matching_results_path": str(matching_tsv),
        "candidate_pairs_path": str(candidate_tsv),
    }

    logger.info(f"Inference complete: {num_matched} entities matched, {num_singletons} singletons.")
    logger.info(f"Output saved to {matching_tsv} and {candidate_tsv}")
    return report


def _write_empty_submission(config: Dict[str, Any], storage: StorageManager) -> None:
    """Write empty but structurally valid TSVs when no candidates exist."""
    sep = config["dataset"].get("separator", "\t")
    enc = config["dataset"].get("encoding", "utf-8")
    s1_test_file = storage.raw_dir / config["dataset"]["files"].get("test_source1", "test_source1.tsv")

    all_s1_ids: List[str] = []
    if s1_test_file.exists():
        with open(s1_test_file, "r", encoding=enc, errors="replace") as f:
            reader = csv.reader(f, delimiter=sep)
            next(reader, None)
            for row in reader:
                if row:
                    all_s1_ids.append(row[0].strip())

    storage.output_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
        col = "matching_ids" if "matching" in fname else "candidate_ids"
        with open(storage.output_dir / fname, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["s1_entity_id", col])
            for s1 in all_s1_ids:
                writer.writerow([s1, ""])
