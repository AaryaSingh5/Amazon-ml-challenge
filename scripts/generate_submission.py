#!/usr/bin/env python3
"""Submission generation CLI script."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.stage_runner import StageController


def main():
    controller = StageController()
    controller.run_stage("test_inference")
    controller.run_stage("submission")


if __name__ == "__main__":
    main()
