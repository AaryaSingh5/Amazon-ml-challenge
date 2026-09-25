"""Stage 13: Final Model Selection.

Reads all experiment metrics from the experiments/ directory, compares
macro F0.5 scores, confirms the best model checkpoint, and writes a
structured final_model_selection.json artifact.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


def run_model_selection(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    """Selects best model checkpoint based on fixed validation macro F0.5."""
    logger.info("Starting Stage 13: Model Selection...")
    storage.initialize_directories()

    # 1. Collect all experiment metrics from experiments/ directory
    experiments_dir = storage.experiments_dir
    all_experiments: List[Dict[str, Any]] = []

    for metrics_file in sorted(experiments_dir.rglob("metrics.json")):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            exp_name = metrics_file.parent.name
            # Normalize to extract best F score
            f_score = None
            if "best_macro_f_beta" in data:
                f_score = data["best_macro_f_beta"]
            elif "metrics" in data:
                m = data["metrics"]
                f_score = m.get("macro_f0.5") or m.get("macro_f_beta")
            if f_score is not None:
                all_experiments.append({
                    "experiment": exp_name,
                    "file": str(metrics_file),
                    "macro_f_beta": float(f_score),
                    "data": data,
                })
        except Exception as e:
            logger.warning(f"Could not read {metrics_file}: {e}")

    if not all_experiments:
        logger.warning("No experiment metrics found. Run Stages 8-11 first.")
        return {"status": "no_experiments"}

    # 2. Sort by F score (descending)
    all_experiments.sort(key=lambda x: x["macro_f_beta"], reverse=True)
    best_exp = all_experiments[0]
    logger.info(f"Best experiment: {best_exp['experiment']} (F={best_exp['macro_f_beta']:.4f})")

    # 3. Determine best model checkpoint path
    best_checkpoint = None
    # Prefer checkpoints/best if it exists (from Stage 11)
    best_dir_model = storage.checkpoints_dir / "best" / "model.joblib"
    latest_dir_model = storage.checkpoints_dir / "latest" / "model.joblib"

    if best_dir_model.exists():
        best_checkpoint = str(best_dir_model)
    elif latest_dir_model.exists():
        best_checkpoint = str(latest_dir_model)

    # 4. Load optimal threshold
    opt_th_file = storage.artifacts_dir / "optimal_thresholds.json"
    optimal_threshold = 0.5
    if opt_th_file.exists():
        with open(opt_th_file, "r") as f:
            optimal_threshold = json.load(f).get("optimal_threshold", 0.5)

    # 5. Build selection report
    report = {
        "selected_experiment": best_exp["experiment"],
        "best_macro_f_beta": best_exp["macro_f_beta"],
        "best_checkpoint_path": best_checkpoint,
        "optimal_threshold": optimal_threshold,
        "all_experiments_ranked": [
            {"experiment": e["experiment"], "macro_f_beta": e["macro_f_beta"]}
            for e in all_experiments
        ],
    }

    # 6. Save
    out_json = storage.artifacts_dir / "final_model_selection.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Final model selection saved to {out_json}")

    # Print leaderboard
    print("\n" + "=" * 60)
    print("  MODEL SELECTION LEADERBOARD")
    print("=" * 60)
    for i, e in enumerate(all_experiments, 1):
        marker = "  <-- SELECTED" if i == 1 else ""
        print(f"  {i}. {e['experiment']:<30} F0.5={e['macro_f_beta']:.4f}{marker}")
    print(f"\n  Optimal Threshold : {optimal_threshold}")
    print(f"  Best Checkpoint   : {best_checkpoint}")
    print("=" * 60 + "\n")

    return report
