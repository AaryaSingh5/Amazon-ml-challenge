"""Stage 5: Candidate Generation and Blocking Pipeline.

Coordinates multi-strategy blocking across Source 1, Source 2, and Source 3 records,
enforcing deduplication, S1->(S2|S3) matching, and candidate capacity limits.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from src.utils.storage import StorageManager
from src.blocking.rules import BlockingEngine

logger = logging.getLogger(__name__)


def load_table_records(file_path: Path, sep: str = "\t", encoding: str = "utf-8") -> List[Dict[str, str]]:
    """Loads records from Parquet or TSV table."""
    records: List[Dict[str, str]] = []
    
    # Try Parquet first if pandas is available
    if file_path.suffix == ".parquet" and file_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            return df.to_dict(orient="records")
        except Exception:
            pass

    # Fallback to TSV
    tsv_path = file_path.with_suffix(".tsv") if file_path.suffix == ".parquet" else file_path
    if not tsv_path.exists():
        alt = file_path.parent / (file_path.stem + ".tsv")
        if alt.exists():
            tsv_path = alt

    if tsv_path.exists():
        with open(tsv_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                if row and "entity_id" in row:
                    records.append(row)

    return records


def run_blocking(
    config: Dict[str, Any],
    storage: StorageManager,
    is_test: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Executes Stage 5 candidate generation pipeline."""
    logger.info(f"Starting Stage 5: Candidate Generation (Mode: {'TEST' if is_test else 'TRAIN'})...")
    storage.initialize_directories()

    proc_dir = storage.processed_dir
    cand_dir = storage.candidates_dir
    cand_dir.mkdir(parents=True, exist_ok=True)

    out_stem = "test_candidates" if is_test else "train_candidates"
    out_parquet = cand_dir / f"{out_stem}.parquet"
    out_tsv = cand_dir / f"{out_stem}.tsv"

    # Check caching
    if not force and out_parquet.exists() and out_parquet.stat().st_size > 0:
        logger.info(f"Reusing cached candidate pairs: {out_parquet}")
        return {"status": "cached", "parquet_path": str(out_parquet), "tsv_path": str(out_tsv)}

    prefix = "test_" if is_test else "train_"
    s1_path = proc_dir / f"{prefix}source1_clean.parquet"
    s2_path = proc_dir / f"{prefix}source2_clean.parquet"
    s3_path = proc_dir / f"{prefix}source3_clean.parquet"

    # Load cleaned records
    s1_recs = load_table_records(s1_path)
    s2_recs = load_table_records(s2_path)
    s3_recs = load_table_records(s3_path)

    if not s1_recs or (not s2_recs and not s3_recs):
        logger.warning(f"Clean records not found in {proc_dir}. Please run Stage 3 Preprocessing first.")
        return {"status": "missing_inputs", "num_candidates": 0}

    logger.info(f"Loaded {len(s1_recs):,} S1, {len(s2_recs):,} S2, {len(s3_recs):,} S3 records.")

    # Initialize blocking engine
    max_cands = config.get("blocking", {}).get("max_candidates_per_entity", 100)
    engine = BlockingEngine(max_candidates_per_entity=max_cands)
    logger.info("Indexing candidate target pool (S2 & S3)...")
    engine.fit_targets(s2_records=s2_recs, s3_records=s3_recs)

    # Generate candidate pairs
    candidate_pairs: List[Dict[str, Any]] = []
    total_candidates = 0
    cands_per_s1: List[int] = []

    # Map for fast candidate output TSV (s1_id \t cand1,cand2,...)
    entity_cand_map: Dict[str, List[str]] = {}

    for s1_rec in s1_recs:
        s1_id = s1_rec["entity_id"].strip()
        cands_with_rules = engine.generate_candidates_for_s1(s1_rec)
        
        target_ids = [t_id for t_id, _ in cands_with_rules]
        entity_cand_map[s1_id] = target_ids
        cands_per_s1.append(len(target_ids))
        total_candidates += len(target_ids)

        for target_id, rules in cands_with_rules:
            candidate_pairs.append({
                "s1_entity_id": s1_id,
                "target_entity_id": target_id,
                "target_source": "S2" if ("S2" in target_id.upper()) else "S3",
                "rules_triggered": ",".join(sorted(rules)),
                "num_rules_triggered": len(rules),
            })

    logger.info(f"Generated {total_candidates:,} candidate pairs for {len(s1_recs):,} S1 entities.")
    mean_cands = (total_candidates / len(s1_recs)) if s1_recs else 0.0
    logger.info(f"Mean candidates per entity: {mean_cands:.2f} (Max: {max(cands_per_s1) if cands_per_s1 else 0})")

    # 1. Save Pairwise Dataset
    try:
        import pandas as pd
        df = pd.DataFrame(candidate_pairs)
        df.to_parquet(out_parquet, index=False)
        logger.info(f"Saved candidate pairs to Parquet: {out_parquet}")
    except Exception:
        pass

    # 2. Save Pairwise TSV
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["s1_entity_id", "target_entity_id", "target_source", "rules_triggered", "num_rules_triggered"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(candidate_pairs)

    # 3. Save Challenge-Format candidate_pairs.tsv (for validation/submission format)
    formatted_cand_tsv = storage.output_dir / ("test_candidate_pairs.tsv" if is_test else "train_candidate_pairs.tsv")
    formatted_cand_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(formatted_cand_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["entity_id", "candidate_entity_ids"])
        for s1_rec in s1_recs:
            s1_id = s1_rec["entity_id"].strip()
            c_list = entity_cand_map.get(s1_id, [])
            writer.writerow([s1_id, ",".join(c_list)])
    logger.info(f"Saved challenge format candidate pairs to {formatted_cand_tsv}")

    # Summary manifest
    summary = {
        "is_test": is_test,
        "total_s1_entities": len(s1_recs),
        "total_candidate_pairs": total_candidates,
        "mean_candidates_per_entity": round(mean_cands, 2),
        "max_candidates_per_entity": max(cands_per_s1) if cands_per_s1 else 0,
        "parquet_path": str(out_parquet),
        "tsv_path": str(out_tsv),
    }

    manifest_path = cand_dir / f"{out_stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
