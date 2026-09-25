"""Checkpoint management system.

Maintains atomic checkpoints with metadata (Git commit, configuration, validation score,
random seed, feature versions) across latest, best, and experiment directories.
"""

from __future__ import annotations

import os
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import joblib
except ImportError:
    joblib = None

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash or 'unknown' if not in git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unversioned"


class CheckpointManager:
    """Handles saving, loading, ranking, and resuming model checkpoints."""

    def __init__(self, checkpoints_root: str | Path = "checkpoints", higher_is_better: bool = True):
        self.root = Path(checkpoints_root)
        self.latest_dir = self.root / "latest"
        self.best_dir = self.root / "best"
        self.experiments_dir = self.root / "experiments"
        self.higher_is_better = higher_is_better

        self._init_dirs()

    def _init_dirs(self) -> None:
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: Any,
        metric_value: float,
        metric_name: str = "macro_f0.5",
        epoch: int = 0,
        global_step: int = 0,
        config: Optional[Dict[str, Any]] = None,
        extra_artifacts: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        feature_version: str = "v1",
        preprocessing_version: str = "v1",
        validation_split_id: str = "val_default",
    ) -> Dict[str, Any]:
        """Saves checkpoint to 'latest' and optionally updates 'best' and 'experiments'."""
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": get_git_commit_hash(),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "epoch": epoch,
            "global_step": global_step,
            "feature_version": feature_version,
            "preprocessing_version": preprocessing_version,
            "validation_split_id": validation_split_id,
            "config": config or {},
        }

        # 1. Save to latest
        self._write_checkpoint_dir(self.latest_dir, model, metadata, extra_artifacts)
        logger.info(f"[Checkpoint] Saved latest checkpoint (epoch {epoch}, {metric_name}: {metric_value:.4f})")

        # 2. Check if this is the best
        is_best = False
        best_meta_path = self.best_dir / "metadata.json"
        if not best_meta_path.exists():
            is_best = True
        else:
            with open(best_meta_path, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
            prev_best = best_meta.get("metric_value", -float("inf") if self.higher_is_better else float("inf"))
            if (self.higher_is_better and metric_value > prev_best) or (not self.higher_is_better and metric_value < prev_best):
                is_best = True

        if is_best:
            self._write_checkpoint_dir(self.best_dir, model, metadata, extra_artifacts)
            logger.info(f"[Checkpoint] *** New Best Model Recorded! ({metric_name}: {metric_value:.4f}) ***")

        # 3. Save to experiment archive if provided
        if experiment_id:
            exp_dir = self.experiments_dir / experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            self._write_checkpoint_dir(exp_dir, model, metadata, extra_artifacts)

        return metadata

    def _write_checkpoint_dir(
        self,
        target_dir: Path,
        model: Any,
        metadata: Dict[str, Any],
        extra_artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Save model
        model_path = target_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save extra artifacts (scalers, encoders, threshold dicts)
        if extra_artifacts:
            for name, obj in extra_artifacts.items():
                artifact_path = target_dir / f"{name}.joblib"
                joblib.dump(obj, artifact_path)

        # Save metadata
        meta_path = target_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Loads the latest checkpoint."""
        return self._load_from_dir(self.latest_dir)

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Loads the best checkpoint."""
        return self._load_from_dir(self.best_dir)

    def _load_from_dir(self, directory: Path) -> Optional[Dict[str, Any]]:
        model_path = directory / "model.joblib"
        meta_path = directory / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model = joblib.load(model_path)
        
        # Load any extra artifacts
        extra_artifacts = {}
        for p in directory.glob("*.joblib"):
            if p.name != "model.joblib":
                extra_artifacts[p.stem] = joblib.load(p)

        return {
            "model": model,
            "metadata": metadata,
            "extra_artifacts": extra_artifacts,
            "directory": str(directory),
        }
