#!/usr/bin/env python3
"""Environment setup and verification script for Kaggle / Colab / Local runtimes.

Verifies GPU acceleration, dependencies, repository state, storage connectivity,
and automatically extracts the Kaggle input zip file if present.
"""

from __future__ import annotations

import os
import sys
import shutil
import zipfile
import platform
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.storage import StorageManager
from src.utils.checkpoints import get_git_commit_hash


def extract_kaggle_dataset(zip_path: str | Path, target_dir: str | Path) -> None:
    """Extracts the Kaggle input zip file to the raw data directory if needed."""
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if raw dataset files are already extracted
    existing_files = list(target_dir.glob("*.tsv"))
    if existing_files:
        print(f"Dataset already extracted in {target_dir} ({len(existing_files)} TSV files found).")
        return

    if zip_path.exists():
        print(f"Extracting dataset from {zip_path} to {target_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("Dataset extraction completed successfully.")
    else:
        print(f"Warning: Dataset zip file not found at {zip_path}")


def print_system_info():
    print("=" * 80)
    print("          SYSTEM & RUNTIME ENVIRONMENT AUDIT          ")
    print("=" * 80)
    print(f"OS Platform     : {platform.platform()}")
    print(f"Python Version  : {sys.version.split()[0]}")
    print(f"Project Root    : {PROJECT_ROOT}")
    print(f"Git Commit Hash : {get_git_commit_hash()}")

    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"PyTorch Version : {torch.__version__}")
        print(f"CUDA Available  : {gpu_available}")
        if gpu_available:
            print(f"GPU Device Name : {torch.cuda.get_device_name(0)}")
            print(f"Device Count    : {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch is not yet installed in this environment.")

    # Initialize Storage Paths
    storage = StorageManager()
    storage.initialize_directories()

    print(f"DATA_ROOT       : {storage.data_root.resolve()}")
    print(f"STORAGE_ROOT    : {storage.storage_root.resolve()}")
    print(f"RAW_DIR         : {storage.raw_dir.resolve()}")
    print(f"Master Zip Path : {storage.gdrive_zip_path}")
    print("=" * 80 + "\n")

    # Automatically extract Kaggle input zip if available
    extract_kaggle_dataset(
        zip_path=storage.gdrive_zip_path,
        target_dir=storage.raw_dir
    )


def setup_environment(install_deps: bool = False):
    if install_deps:
        req_file = PROJECT_ROOT / "requirements.txt"
        if req_file.exists():
            print(f"Installing dependencies from {req_file}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)], check=True)
            print("Dependencies installed.")

    print_system_info()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup Kaggle/Local environment")
    parser.add_argument("--install", action="store_true", help="Install requirements.txt")
    args = parser.parse_args()
    setup_environment(install_deps=args.install)