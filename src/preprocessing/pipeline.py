"""Stage 3: Normalization and Preprocessing Pipeline.

Loads raw business entity tables, generates multi-representation clean fields,
and serializes reusable processed parquet/tsv artifacts without data loss.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.utils.storage import StorageManager
from src.preprocessing.normalizer import transform_record

logger = logging.getLogger(__name__)


def process_table_file(
    input_path: Path,
    output_parquet_path: Path,
    output_csv_path: Optional[Path] = None,
    sep: str = "\t",
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Processes a single raw business table and saves multi-representation clean records."""
    if not input_path.exists() or input_path.stat().st_size == 0:
        logger.warning(f"File not found or empty: {input_path}")
        return {"status": "file_not_found", "path": str(input_path)}

    logger.info(f"Processing table: {input_path.name}...")
    records: List[Dict[str, str]] = []
    header: List[str] = []

    with open(input_path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader, None)
        if not header:
            return {"status": "empty_table", "num_rows": 0}

        col_idx = {c.strip().lower(): i for i, c in enumerate(header)}
        ent_i = col_idx.get("entity_id", 0)
        name_i = col_idx.get("business_name", 1 if len(header) > 1 else 0)
        addr_i = col_idx.get("business_address", 2 if len(header) > 2 else 0)
        cntry_i = col_idx.get("country", 3 if len(header) > 3 else 0)

        for row in reader:
            if not row:
                continue
            e_id = row[ent_i] if ent_i < len(row) else ""
            b_name = row[name_i] if name_i < len(row) else ""
            b_addr = row[addr_i] if addr_i < len(row) else ""
            b_cntry = row[cntry_i] if cntry_i < len(row) else ""

            clean_rec = transform_record(
                entity_id=e_id,
                business_name=b_name,
                business_address=b_addr,
                country=b_cntry,
            )
            records.append(clean_rec)

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    num_records = len(records)
    saved_parquet = False

    # Try saving with pandas / pyarrow if installed
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_parquet(output_parquet_path, index=False, engine="auto")
        saved_parquet = True
        logger.info(f"Saved {num_records:,} clean records to Parquet: {output_parquet_path}")
    except Exception as e:
        logger.warning(f"Could not save Parquet (pandas/pyarrow not present or failed: {e}). Fallback to TSV.")

    # Always write TSV/CSV as well for universal compatibility
    tsv_out = output_parquet_path.with_suffix(".tsv")
    fieldnames = [
        "entity_id",
        "original_name",
        "normalized_name",
        "clean_name_no_suffix",
        "original_address",
        "normalized_address",
        "postal_code",
        "address_digits",
        "original_country",
        "normalized_country",
    ]
    with open(tsv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {num_records:,} clean records to TSV: {tsv_out}")

    return {
        "status": "success",
        "num_records": num_records,
        "parquet_path": str(output_parquet_path) if saved_parquet else None,
        "tsv_path": str(tsv_out),
    }


def run_preprocessing(
    config: Dict[str, Any],
    storage: StorageManager,
    force: bool = False,
) -> Dict[str, Any]:
    """Executes Stage 3 Preprocessing across all training and test tables."""
    logger.info("Starting Stage 3: Normalization & Preprocessing Pipeline...")
    storage.initialize_directories()

    raw_dir = storage.raw_dir
    proc_dir = storage.processed_dir
    file_map = config["dataset"]["files"]
    sep = config["dataset"].get("separator", "\t")
    encoding = config["dataset"].get("encoding", "utf-8")

    from datetime import datetime
    manifest: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "processed_tables": {},
    }

    # Process all source tables
    tables_to_process = [
        ("train_source1", "train_source1_clean"),
        ("train_source2", "train_source2_clean"),
        ("train_source3", "train_source3_clean"),
        ("test_source1", "test_source1_clean"),
        ("test_source2", "test_source2_clean"),
        ("test_source3", "test_source3_clean"),
    ]

    for key, out_stem in tables_to_process:
        fname = file_map.get(key, f"{key}.tsv")
        in_path = raw_dir / fname
        if not in_path.exists():
            cands = list(raw_dir.glob(f"*{key}*"))
            if cands:
                in_path = cands[0]

        if in_path.exists() and in_path.stat().st_size > 0:
            out_parquet = proc_dir / f"{out_stem}.parquet"
            # Check caching
            if not force and out_parquet.exists() and out_parquet.stat().st_size > 0:
                logger.info(f"Reusing cached clean table: {out_parquet}")
                manifest["processed_tables"][key] = {"status": "cached", "path": str(out_parquet)}
            else:
                res = process_table_file(in_path, out_parquet, sep=sep, encoding=encoding)
                manifest["processed_tables"][key] = res
        else:
            logger.info(f"Skipping table {key} (not present in raw dir).")
            manifest["processed_tables"][key] = {"status": "not_present"}

    # Save preprocessing metadata manifest
    manifest_path = proc_dir / "preprocessing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Preprocessing manifest saved to {manifest_path}")

    return manifest
