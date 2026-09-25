"""Stage Controller and Pipeline Orchestrator.

Manages stage transitions, dependency validation, artifact verification,
and independent step execution with explicit caching controls.
"""

from __future__ import annotations

import os
import sys
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import yaml

from src.utils.storage import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("StageRunner")


class StageController:
    """Controls execution of pipeline stages with artifact caching and dependency checks."""

    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.storage = StorageManager(config_path)
        self.storage.initialize_directories()
        
        self.stage_defs = self.config["stages"]["definitions"]
        self.cache_enabled = self.config["stages"].get("cache_artifacts", True)

    def print_banner(self, stage_name: str) -> None:
        """Prints a prominent visual banner for the current stage."""
        stage_info = self.stage_defs.get(stage_name, {})
        stage_id = stage_info.get("id", "STAGE")
        disp_name = stage_info.get("name", stage_name.upper())

        sep = "=" * 80
        print("\n" + sep)
        print(f" >>> RUNNING: {stage_id} - {disp_name} <<<")
        print(sep)
        print(f"Config: {self.config_path}")
        print(f"Data Root: {self.storage.data_root}")
        print(f"Storage Root: {self.storage.storage_root}")
        print(sep + "\n")

    def check_inputs(self, stage_name: str) -> bool:
        """Verifies all required input artifacts for the requested stage."""
        stage_info = self.stage_defs.get(stage_name)
        if not stage_info:
            logger.error(f"Unknown stage: '{stage_name}'. Available: {list(self.stage_defs.keys())}")
            return False

        inputs = stage_info.get("required_inputs", [])
        missing = []
        for inp in inputs:
            p = Path(inp)
            if not self.storage.artifact_exists(p):
                missing.append(inp)

        if missing:
            print("\n" + "!" * 80)
            logger.error(f"Cannot execute '{stage_name}'. Missing prerequisite artifacts:")
            for m in missing:
                logger.error(f"  - MISSING: {m}")
            print("\nPlease execute prerequisite earlier stages first.")
            print("!" * 80 + "\n")
            return False
        return True

    def is_stage_cached(self, stage_name: str) -> bool:
        """Checks if all outputs of this stage already exist."""
        stage_info = self.stage_defs.get(stage_name)
        if not stage_info:
            return False
        outputs = stage_info.get("output_artifacts", [])
        if not outputs:
            return False
        for out in outputs:
            if not self.storage.artifact_exists(out):
                return False
        return True

    def run_stage(
        self,
        stage_name: str,
        stage_func: Optional[Callable[..., Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Executes a single stage with dependency checks and caching."""
        stage_name = stage_name.lower().strip()
        if stage_name not in self.stage_defs:
            logger.error(f"Invalid stage '{stage_name}'. Valid stages: {list(self.stage_defs.keys())}")
            return False

        self.print_banner(stage_name)

        # 1. Dependency validation
        if not self.check_inputs(stage_name):
            return False

        # 2. Cache check
        if not force and self.cache_enabled and self.is_stage_cached(stage_name):
            logger.info(f"Stage '{stage_name}' output artifacts already exist. Reusing cached artifacts.")
            stage_info = self.stage_defs[stage_name]
            for out in stage_info.get("output_artifacts", []):
                logger.info(f"  - Existing artifact: {out}")
            logger.info(f"To recompute, run with --force")
            return True

        # 3. Stage Execution
        try:
            logger.info(f"Executing stage logic for '{stage_name}'...")
            if stage_func:
                stage_func(self.config, self.storage, **kwargs)
            else:
                # Try dynamic dispatch if module exists
                self._dispatch_stage_script(stage_name)

            logger.info(f"*** Stage '{stage_name}' completed successfully! ***")
            return True
        except Exception as e:
            logger.exception(f"Error occurred while executing stage '{stage_name}': {e}")
            return False

    def _dispatch_stage_script(self, stage_name: str) -> None:
        """Dynamically imports and executes corresponding stage function."""
        mapping = {
            "data_audit": ("src.data.audit", "run_data_audit"),
            "eda": ("src.data.eda", "run_eda"),
            "preprocessing": ("src.preprocessing.pipeline", "run_preprocessing"),
            "validation": ("src.evaluation.split", "run_validation_split"),
            "blocking": ("src.blocking.pipeline", "run_blocking"),
            "candidate_recall": ("src.evaluation.candidate_recall", "run_candidate_recall"),
            "features": ("src.features.pipeline", "run_features"),
            "baseline": ("src.models.baseline", "run_baseline"),
            "training": ("src.training.train", "run_training"),
            "threshold_optimization": ("src.evaluation.threshold_optimizer", "run_threshold_optimization"),
            "advanced_modeling": ("src.training.advanced", "run_advanced_training"),
            "hard_negatives": ("src.features.hard_negatives", "run_hard_negatives"),
            "model_selection": ("src.models.selection", "run_model_selection"),
            "test_inference": ("src.inference.predict", "run_inference"),
            "submission": ("src.submission.package", "run_submission_packaging"),
            "reproducibility_package": ("src.submission.reproducibility", "run_reproducibility_package"),
        }
        if stage_name not in mapping:
            raise NotImplementedError(f"No runner defined yet for stage '{stage_name}'.")

        mod_path, func_name = mapping[stage_name]
        module = importlib.import_module(mod_path)
        func = getattr(module, func_name)
        func(self.config, self.storage)
