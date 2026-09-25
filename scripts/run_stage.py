#!/usr/bin/env python3
"""CLI entrypoint to run individual pipeline stages.

Usage:
    python scripts/run_stage.py --stage data_audit
    python scripts/run_stage.py --stage preprocessing --force
    python scripts/run_stage.py --stage baseline --config configs/config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.stage_runner import StageController


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge - Entity Resolution Stage Runner")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Stage to execute (e.g. data_audit, eda, preprocessing, validation, blocking, features, baseline, training, ...)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output artifacts exist",
    )

    args = parser.parse_args()

    controller = StageController(config_path=args.config)
    success = controller.run_stage(stage_name=args.stage, force=args.force)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
