"""Storage and path management abstraction.

Provides a unified interface for data access, synchronization between Google Drive
and local Kaggle/development runtimes, artifact caching, and directory management.
"""

from __future__ import annotations

import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage paths, Google Drive synchronization, and artifact verification."""

    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Determine roots from environment variables or config
        self.data_root = Path(os.environ.get("DATA_ROOT", self.config["storage"]["data_root"]))
        self.storage_root = Path(os.environ.get("STORAGE_ROOT", self.config["storage"]["storage_root"]))
        
        # Subdirectories
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.splits_dir = self.data_root / "splits"
        self.candidates_dir = self.data_root / "candidates"
        self.features_dir = self.data_root / "features"
        
        self.checkpoints_dir = self.storage_root / "checkpoints"
        self.experiments_dir = self.storage_root / "experiments"
        self.artifacts_dir = self.storage_root / "artifacts"
        self.output_dir = self.storage_root / "output"
        self.logs_dir = self.storage_root / "logs"

        # Raw file names
        self.file_names = self.config["dataset"]["files"]
        self.gdrive_zip_path = Path(self.config["storage"].get("gdrive_dataset_zip", ""))

    def _load_config(self) -> Dict[str, Any]:
        """Loads the master YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def initialize_directories(self) -> None:
        """Ensures all necessary runtime and persistent directories exist."""
        dirs = [
            self.data_root,
            self.raw_dir,
            self.processed_dir,
            self.splits_dir,
            self.candidates_dir,
            self.features_dir,
            self.storage_root,
            self.checkpoints_dir,
            self.checkpoints_dir / "latest",
            self.checkpoints_dir / "best",
            self.experiments_dir,
            self.artifacts_dir,
            self.output_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Storage directories initialized successfully.")

    def sync_dataset_from_zip(self, zip_path: Optional[str | Path] = None, force: bool = False) -> bool:
        """Syncs / extracts dataset from a zip file (e.g. from Google Drive or local path)."""
        target_zip = Path(zip_path) if zip_path else self.gdrive_zip_path
        
        # Check if raw files already exist and are populated
        if not force and self.has_raw_dataset():
            logger.info(f"Raw dataset already exists in {self.raw_dir}. Skipping extraction.")
            return True

        if not target_zip.exists():
            # Try looking in alternative common locations
            alt_locations = [
                Path("/content") / target_zip,
                Path("/kaggle/input") / target_zip.name,
                Path("data") / target_zip.name,
                Path(".") / target_zip.name,
            ]
            found = False
            for alt in alt_locations:
                if alt.exists():
                    target_zip = alt
                    found = True
                    break
            if not found:
                logger.warning(
                    f"Master dataset zip not found at {target_zip}. "
                    f"Please mount Google Drive or ensure dataset is available at {self.gdrive_zip_path}"
                )
                return False

        logger.info(f"Extracting dataset from {target_zip} to {self.raw_dir}...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            # Extract and flatten if needed
            for member in zip_ref.infolist():
                filename = Path(member.filename).name
                if filename.endswith(".tsv") or filename.endswith(".csv"):
                    target_file = self.raw_dir / filename
                    with zip_ref.open(member) as source, open(target_file, "wb") as target:
                        shutil.copyfileobj(source, target)
                        logger.info(f"Extracted: {filename} -> {target_file}")
                elif not member.is_dir():
                    # Extract general files
                    zip_ref.extract(member, self.raw_dir)
                    
        return self.has_raw_dataset()

    def has_raw_dataset(self) -> bool:
        """Checks if the required raw training and test TSVs exist and are non-empty."""
        required_keys = ["train_source1", "train_source2", "train_source3"]
        for key in required_keys:
            fname = self.file_names.get(key)
            if not fname:
                continue
            fpath = self.raw_dir / fname
            if not fpath.exists() or fpath.stat().st_size == 0:
                return False
        return True

    def get_raw_file_path(self, file_key: str) -> Path:
        """Returns the full path to a raw dataset file."""
        fname = self.file_names.get(file_key, file_key)
        return self.raw_dir / fname

    def artifact_exists(self, relative_path: str | Path) -> bool:
        """Checks if an artifact exists and is non-empty."""
        p = Path(relative_path)
        if not p.is_absolute():
            # Check relative to repo root or storage root
            p1 = Path(relative_path)
            p2 = self.storage_root / relative_path
            p3 = self.data_root / relative_path
            for candidate in [p1, p2, p3]:
                if candidate.exists() and candidate.stat().st_size > 0:
                    return True
            return False
        return p.exists() and p.stat().st_size > 0


# Global helper instance
storage_manager = StorageManager()
