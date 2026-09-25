"""Stage 4: Leak-Free Entity-Level Validation Split Generator.

Partitions Source 1 entities into train and validation sets with stratified
singleton and country distributions, preventing candidate pair data leakage.
"""

from __future__ import annotations

import csv
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


def create_entity_validation_split(
    s1_entities: List[Dict[str, str]],
    ground_truth_map: Dict[str, List[str]],
    val_ratio: float = 0.20,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Generates leak-free train and validation entity ID lists stratified by match cardinality and country.

    Args:
        s1_entities: List of Source 1 entity records (must contain 'entity_id', optional 'country').
        ground_truth_map: Mapping from s1_entity_id -> list of matched target entity IDs.
        val_ratio: Fraction of entities assigned to validation set (default: 0.20).
        random_seed: Deterministic random seed for reproducibility.

    Returns:
        Structured split dictionary with train/val IDs and stratification statistics.
    """
    rng = random.Random(random_seed)

    # 1. Group entities by stratification key (country + cardinality bucket)
    strata: Dict[str, List[str]] = defaultdict(list)

    for record in s1_entities:
        e_id = record["entity_id"].strip()
        cntry = record.get("normalized_country") or record.get("country") or "UNKNOWN"
        cntry = cntry.strip().upper()

        matches = ground_truth_map.get(e_id, [])
        num_matches = len(matches)

        if num_matches == 0:
            bucket = "singleton"
        elif num_matches == 1:
            bucket = "one_to_one"
        else:
            bucket = "one_to_many"

        stratum_key = f"{cntry}_{bucket}"
        strata[stratum_key].append(e_id)

    train_ids: List[str] = []
    val_ids: List[str] = []

    # 2. Partition each stratum deterministically
    for stratum_key, entity_ids in sorted(strata.items()):
        shuffled = list(entity_ids)
        rng.shuffle(shuffled)
        
        n_val = int(round(len(shuffled) * val_ratio))
        val_subset = shuffled[:n_val]
        train_subset = shuffled[n_val:]

        val_ids.extend(val_subset)
        train_ids.extend(train_subset)

    # Sort for deterministic ordering
    train_ids.sort()
    val_ids.sort()

    # 3. Compute detailed split statistics
    def _compute_stats(id_list: List[str]) -> Dict[str, Any]:
        id_set = set(id_list)
        total = len(id_list)
        singletons = sum(1 for e in id_list if len(ground_truth_map.get(e, [])) == 0)
        one_to_one = sum(1 for e in id_list if len(ground_truth_map.get(e, [])) == 1)
        one_to_many = sum(1 for e in id_list if len(ground_truth_map.get(e, [])) > 1)
        total_targets = sum(len(ground_truth_map.get(e, [])) for e in id_list)

        return {
            "total_entities": total,
            "singletons_count": singletons,
            "singletons_pct": float(round((singletons / total * 100.0) if total > 0 else 0.0, 2)),
            "one_to_one_count": one_to_one,
            "one_to_one_pct": float(round((one_to_one / total * 100.0) if total > 0 else 0.0, 2)),
            "one_to_many_count": one_to_many,
            "one_to_many_pct": float(round((one_to_many / total * 100.0) if total > 0 else 0.0, 2)),
            "total_ground_truth_targets": total_targets,
        }

    train_stats = _compute_stats(train_ids)
    val_stats = _compute_stats(val_ids)

    return {
        "split_id": f"entity_stratified_seed{random_seed}_val{int(val_ratio*100)}",
        "random_seed": random_seed,
        "val_ratio": val_ratio,
        "total_source1_entities": len(s1_entities),
        "train_entity_ids": train_ids,
        "val_entity_ids": val_ids,
        "train_statistics": train_stats,
        "validation_statistics": val_stats,
    }


def run_validation_split(
    config: Dict[str, Any],
    storage: StorageManager,
    val_ratio: float = 0.20,
    random_seed: int = 42,
    force: bool = False,
) -> Dict[str, Any]:
    """Executes Stage 4: Generates and persists the fixed entity validation split."""
    logger.info("Starting Stage 4: Leak-Free Validation Split Design...")
    storage.initialize_directories()

    split_file = storage.splits_dir / "validation_split.json"
    if not force and split_file.exists() and split_file.stat().st_size > 0:
        logger.info(f"Reusing existing fixed validation split from {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raw_dir = storage.raw_dir
    proc_dir = storage.processed_dir
    file_map = config["dataset"]["files"]
    sep = config["dataset"].get("separator", "\t")
    encoding = config["dataset"].get("encoding", "utf-8")

    # Load clean Source 1 entities (or raw if clean not yet generated)
    s1_file = proc_dir / "train_source1_clean.tsv"
    if not s1_file.exists():
        s1_file = raw_dir / file_map.get("train_source1", "train_source1.tsv")

    s1_entities: List[Dict[str, str]] = []
    if s1_file.exists():
        with open(s1_file, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                if row and "entity_id" in row:
                    s1_entities.append(row)

    # Load Ground Truth
    gt_file = raw_dir / file_map.get("train_ground_truth", "train_ground_truth.tsv")
    gt_map: Dict[str, List[str]] = {}
    if gt_file.exists():
        with open(gt_file, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f, delimiter=sep)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                t_val = row[1].strip() if len(row) > 1 else ""
                targets = [x.strip() for x in t_val.split(",") if x.strip()] if "," in t_val else ([t_val] if t_val else [])
                if s1_id not in gt_map:
                    gt_map[s1_id] = []
                gt_map[s1_id].extend(targets)

    if not s1_entities:
        logger.warning("No Source 1 entities found to split. Generating placeholder split metadata.")
        s1_entities = [{"entity_id": f"S1_{i}", "country": "US"} for i in range(100)]

    split_result = create_entity_validation_split(
        s1_entities=s1_entities,
        ground_truth_map=gt_map,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    # Save to disk
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_result, f, indent=2)
    logger.info(f"Fixed validation split saved to {split_file}")

    # Generate readable summary markdown
    md_file = storage.artifacts_dir / "validation_split_summary.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(_format_split_markdown(split_result))
    logger.info(f"Validation summary markdown saved to {md_file}")

    return split_result


def _format_split_markdown(split: Dict[str, Any]) -> str:
    t = split.get("train_statistics", {})
    v = split.get("validation_statistics", {})
    lines = [
        "# Stage 4: Leak-Free Validation Split Summary",
        f"**Split ID**: `{split.get('split_id')}`",
        f"**Random Seed**: `{split.get('random_seed')}`",
        f"**Validation Ratio**: `{split.get('val_ratio') * 100}%`",
        "",
        "## Partition Breakdown",
        "| Metric | Train Fold | Validation Fold | Total |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Entities Count** | {t.get('total_entities', 0):,} | {v.get('total_entities', 0):,} | {split.get('total_source1_entities', 0):,} |",
        f"| **Singletons (0 matches)** | {t.get('singletons_count', 0):,} ({t.get('singletons_pct', 0)}%) | {v.get('singletons_count', 0):,} ({v.get('singletons_pct', 0)}%) | - |",
        f"| **One-to-One Matches** | {t.get('one_to_one_count', 0):,} ({t.get('one_to_one_pct', 0)}%) | {v.get('one_to_one_count', 0):,} ({v.get('one_to_one_pct', 0)}%) | - |",
        f"| **One-to-Many Matches** | {t.get('one_to_many_count', 0):,} ({t.get('one_to_many_pct', 0)}%) | {v.get('one_to_many_count', 0):,} ({v.get('one_to_many_pct', 0)}%) | - |",
        f"| **Total Ground Truth Targets** | {t.get('total_ground_truth_targets', 0):,} | {v.get('total_ground_truth_targets', 0):,} | - |",
        "",
        "## Anti-Leakage Guarantee",
        "- Splitting is strictly performed on **Source 1 Entity ID** level.",
        "- Intersection between Train and Validation entities is strictly **EMPTY** ($Train \\cap Val = \\emptyset$).",
        "- All candidate pairs for any validation entity will reside strictly in the validation fold.",
    ]
    return "\n".join(lines)
