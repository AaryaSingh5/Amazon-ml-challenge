"""Stage 8: Rule-Based Similarity Baseline Model."""
from __future__ import annotations

import csv
import json
import logging
from typing import Dict, Any, Set, List
from collections import defaultdict
from pathlib import Path

from src.utils.storage import StorageManager
from src.evaluation.metrics import compute_macro_f05

logger = logging.getLogger(__name__)


def run_baseline(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Runs weighted heuristic similarity baseline on candidate pairs."""
    logger.info("Starting Stage 8: Baseline Model...")
    storage.initialize_directories()

    # Load parameters
    name_w = config["models"]["baseline"].get("name_weight", 0.55)
    addr_w = config["models"]["baseline"].get("address_weight", 0.35)
    cntry_w = config["models"]["baseline"].get("country_weight", 0.10)
    threshold = config["models"]["baseline"].get("default_threshold", 0.65)
    beta = config["pipeline"].get("beta", 0.5)

    # 1. Load Validation Split IDs
    val_ids: Set[str] = set()
    split_file = storage.splits_dir / "validation_split.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            val_ids = set(json.load(f).get("val_entity_ids", []))
    else:
        logger.warning(f"Validation split not found at {split_file}. Baseline requires a validation set.")
        return {}

    # 2. Load Ground Truth for Validation Set
    gt_file = storage.raw_dir / config["dataset"]["files"].get("train_ground_truth", "train_ground_truth.tsv")
    gt_map: Dict[str, Set[str]] = {s1: set() for s1 in val_ids}
    if gt_file.exists():
        with open(gt_file, "r", encoding=config["dataset"].get("encoding", "utf-8"), errors="replace") as f:
            reader = csv.reader(f, delimiter=config["dataset"].get("separator", "\t"))
            next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                s1 = row[0].strip()
                if s1 in val_ids:
                    targets = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 else []
                    gt_map[s1].update(targets)

    # 3. Load Features and Predict
    feat_tsv = storage.features_dir / "train_features.tsv"
    if not feat_tsv.exists():
        logger.error(f"Features file not found at {feat_tsv}. Cannot run baseline.")
        return {}

    predictions: Dict[str, Set[str]] = defaultdict(set)
    evaluated_pairs = 0

    with open(feat_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("is_validation") == "1":
                s1 = row["s1_entity_id"]
                t = row["target_entity_id"]
                
                n_tset = float(row.get("name_token_set_ratio", 0.0))
                a_tset = float(row.get("addr_token_set_ratio", 0.0))
                c_exact = float(row.get("country_exact_match", 0.0))

                # Compute weighted score (token_set_ratio is usually 0-100, we normalize to 0-1)
                score = (name_w * (n_tset / 100.0)) + (addr_w * (a_tset / 100.0)) + (cntry_w * c_exact)

                if score >= threshold:
                    predictions[s1].add(t)
                evaluated_pairs += 1

    logger.info(f"Evaluated baseline on {evaluated_pairs} validation candidate pairs.")

    # 4. Compute Metrics
    metrics = compute_macro_f05(ground_truth=gt_map, predictions=predictions, beta=beta)
    
    # 5. Save Results
    exp_dir = storage.experiments_dir / "exp_000_baseline"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_json = exp_dir / "metrics.json"

    report = {
        "model": "rule_based_baseline",
        "threshold": threshold,
        "weights": {
            "name": name_w,
            "address": addr_w,
            "country": cntry_w,
        },
        "metrics": metrics
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Baseline F{beta} Score: {metrics.get(f'macro_f{beta}', 0):.4f}")
    logger.info(f"Baseline evaluation saved to {out_json}")

    return report
