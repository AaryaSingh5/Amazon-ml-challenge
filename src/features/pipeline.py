"""Stage 7: Pairwise Feature Engineering Pipeline.

Computes similarity features across business names, addresses, countries, and
cross-field interactions for all candidate pairs, assigning ground truth labels and fold flags.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

from src.utils.storage import StorageManager
from src.blocking.pipeline import load_table_records
from src.features.similarity import (
    levenshtein_ratio,
    token_sort_ratio,
    token_set_ratio,
    jaccard_similarity,
    char_ngram_jaccard,
    numeric_tokens_jaccard,
    postal_code_agreement,
    length_features,
)

logger = logging.getLogger(__name__)


def extract_pair_features(
    s1_rec: Dict[str, str],
    target_rec: Dict[str, str],
    rules_triggered: str = "",
) -> Dict[str, Any]:
    """Computes all similarity and interaction features for a single candidate pair."""
    # Names
    s1_orig_name = s1_rec.get("original_name", "")
    t_orig_name = target_rec.get("original_name", "")
    s1_norm_name = s1_rec.get("normalized_name", "")
    t_norm_name = target_rec.get("normalized_name", "")
    s1_clean_name = s1_rec.get("clean_name_no_suffix", "")
    t_clean_name = target_rec.get("clean_name_no_suffix", "")

    # Addresses
    s1_orig_addr = s1_rec.get("original_address", "")
    t_orig_addr = target_rec.get("original_address", "")
    s1_norm_addr = s1_rec.get("normalized_address", "")
    t_norm_addr = target_rec.get("normalized_address", "")
    s1_post = s1_rec.get("postal_code", "")
    t_post = target_rec.get("postal_code", "")

    # Countries
    s1_orig_cntry = s1_rec.get("original_country", "")
    t_orig_cntry = target_rec.get("original_country", "")
    s1_norm_cntry = s1_rec.get("normalized_country", "")
    t_norm_cntry = target_rec.get("normalized_country", "")

    # 1. Business Name Features
    n_lev = levenshtein_ratio(s1_norm_name, t_norm_name)
    n_tsort = token_sort_ratio(s1_norm_name, t_norm_name)
    n_tset = token_set_ratio(s1_clean_name or s1_norm_name, t_clean_name or t_norm_name)
    n_jaccard = jaccard_similarity(s1_clean_name or s1_norm_name, t_clean_name or t_norm_name)
    n_3gram = char_ngram_jaccard(s1_clean_name or s1_norm_name, t_clean_name or t_norm_name)
    n_len_diff, n_len_ratio, n_tok_diff = length_features(s1_clean_name, t_clean_name)

    # 2. Address Features
    a_lev = levenshtein_ratio(s1_norm_addr, t_norm_addr)
    a_tsort = token_sort_ratio(s1_norm_addr, t_norm_addr)
    a_tset = token_set_ratio(s1_norm_addr, t_norm_addr)
    a_jaccard = jaccard_similarity(s1_norm_addr, t_norm_addr)
    a_3gram = char_ngram_jaccard(s1_norm_addr, t_norm_addr)
    a_num_overlap = numeric_tokens_jaccard(s1_norm_addr, t_norm_addr)
    a_post_match = postal_code_agreement(s1_post, t_post)
    a_len_diff, a_len_ratio, a_tok_diff = length_features(s1_norm_addr, t_norm_addr)

    # 3. Country Agreement
    c_exact = 1.0 if s1_orig_cntry and t_orig_cntry and (s1_orig_cntry.strip().upper() == t_orig_cntry.strip().upper()) else 0.0
    c_norm = 1.0 if s1_norm_cntry and t_norm_cntry and (s1_norm_cntry == t_norm_cntry) else 0.0

    # 4. Cross-Field & Interactions
    comb_s1 = f"{s1_clean_name} {s1_norm_addr}".strip()
    comb_t = f"{t_clean_name} {t_norm_addr}".strip()
    comb_sim = token_set_ratio(comb_s1, comb_t)
    name_x_addr = n_tset * a_tset
    strong_agree = 1.0 if (n_tset >= 0.85 and a_tset >= 0.80 and c_norm == 1.0) else 0.0

    # 5. Blocking Rule Indicators
    rules_set = set(r.strip() for r in rules_triggered.split(",") if r.strip())
    is_s2 = 1.0 if ("S2" in target_rec.get("entity_id", "").upper()) else 0.0

    return {
        "s1_entity_id": s1_rec.get("entity_id", ""),
        "target_entity_id": target_rec.get("entity_id", ""),
        # Name features
        "name_exact_match": 1.0 if s1_orig_name == t_orig_name and s1_orig_name != "" else 0.0,
        "name_norm_exact_match": 1.0 if s1_norm_name == t_norm_name and s1_norm_name != "" else 0.0,
        "name_clean_exact_match": 1.0 if s1_clean_name == t_clean_name and s1_clean_name != "" else 0.0,
        "name_levenshtein_ratio": round(n_lev, 4),
        "name_token_sort_ratio": round(n_tsort, 4),
        "name_token_set_ratio": round(n_tset, 4),
        "name_jaccard_similarity": round(n_jaccard, 4),
        "name_char_3gram_jaccard": round(n_3gram, 4),
        "name_len_diff_abs": n_len_diff,
        "name_len_ratio": round(n_len_ratio, 4),
        "name_token_count_diff": n_tok_diff,
        # Address features
        "addr_exact_match": 1.0 if s1_orig_addr == t_orig_addr and s1_orig_addr != "" else 0.0,
        "addr_norm_exact_match": 1.0 if s1_norm_addr == t_norm_addr and s1_norm_addr != "" else 0.0,
        "addr_levenshtein_ratio": round(a_lev, 4),
        "addr_token_sort_ratio": round(a_tsort, 4),
        "addr_token_set_ratio": round(a_tset, 4),
        "addr_jaccard_similarity": round(a_jaccard, 4),
        "addr_char_3gram_jaccard": round(a_3gram, 4),
        "addr_numeric_overlap": round(a_num_overlap, 4),
        "addr_postal_pin_match": a_post_match,
        "addr_len_diff_abs": a_len_diff,
        "addr_len_ratio": round(a_len_ratio, 4),
        "addr_token_count_diff": a_tok_diff,
        # Country features
        "country_exact_match": c_exact,
        "country_norm_match": c_norm,
        # Cross-field & Interaction features
        "combined_text_similarity": round(comb_sim, 4),
        "name_x_addr_similarity": round(name_x_addr, 4),
        "strong_agreement_indicator": strong_agree,
        "is_source2": is_s2,
        "num_rules_triggered": len(rules_set),
        "rule_exact_name_flag": 1.0 if "rule_exact_name" in rules_set else 0.0,
        "rule_clean_name_flag": 1.0 if "rule_clean_name" in rules_set else 0.0,
        "rule_exact_addr_flag": 1.0 if "rule_exact_addr" in rules_set else 0.0,
        "rule_prefix_tokens_flag": 1.0 if "rule_prefix_tokens" in rules_set else 0.0,
        "rule_rare_token_flag": 1.0 if "rule_rare_token" in rules_set else 0.0,
        "rule_postal_digits_flag": 1.0 if "rule_postal_digits" in rules_set else 0.0,
        "rule_char_3gram_flag": 1.0 if "rule_char_3gram" in rules_set else 0.0,
    }


def run_features(
    config: Dict[str, Any],
    storage: StorageManager,
    is_test: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Executes Stage 7: Feature Engineering Pipeline."""
    logger.info(f"Starting Stage 7: Feature Engineering (Mode: {'TEST' if is_test else 'TRAIN'})...")
    storage.initialize_directories()

    proc_dir = storage.processed_dir
    cand_dir = storage.candidates_dir
    feat_dir = storage.features_dir
    raw_dir = storage.raw_dir
    splits_dir = storage.splits_dir
    feat_dir.mkdir(parents=True, exist_ok=True)

    out_stem = "test_features" if is_test else "train_features"
    out_parquet = feat_dir / f"{out_stem}.parquet"
    out_tsv = feat_dir / f"{out_stem}.tsv"

    if not force and out_parquet.exists() and out_parquet.stat().st_size > 0:
        logger.info(f"Reusing cached feature matrix: {out_parquet}")
        return {"status": "cached", "parquet_path": str(out_parquet), "tsv_path": str(out_tsv)}

    # 1. Load clean records
    prefix = "test_" if is_test else "train_"
    s1_recs = {r["entity_id"]: r for r in load_table_records(proc_dir / f"{prefix}source1_clean.parquet")}
    s2_recs = {r["entity_id"]: r for r in load_table_records(proc_dir / f"{prefix}source2_clean.parquet")}
    s3_recs = {r["entity_id"]: r for r in load_table_records(proc_dir / f"{prefix}source3_clean.parquet")}
    target_recs = {**s2_recs, **s3_recs}

    # 2. Load candidate pairs
    cand_file = cand_dir / ("test_candidates.tsv" if is_test else "train_candidates.tsv")
    candidate_pairs: List[Dict[str, Any]] = []
    if cand_file.exists():
        with open(cand_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row:
                    candidate_pairs.append(row)

    if not candidate_pairs:
        logger.warning("No candidate pairs available to build features. Run Stage 5 first.")
        return {"status": "no_candidates", "num_pairs": 0}

    # 3. Load Ground Truth and Validation Fold (if training)
    gt_map: Dict[str, Set[str]] = {}
    val_ids: Set[str] = set()

    if not is_test:
        gt_file = raw_dir / config["dataset"]["files"].get("train_ground_truth", "train_ground_truth.tsv")
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    s1 = row[0].strip()
                    targets = [x.strip() for x in row[1].split(",") if x.strip()] if len(row) > 1 else []
                    if s1 not in gt_map:
                        gt_map[s1] = set()
                    gt_map[s1].update(targets)

        split_file = splits_dir / "validation_split.json"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                val_ids = set(s_data.get("val_entity_ids", []))

    # 4. Compute Features for all Candidate Pairs
    logger.info(f"Extracting features for {len(candidate_pairs):,} candidate pairs...")
    feature_rows: List[Dict[str, Any]] = []
    pos_count = 0

    for c_pair in candidate_pairs:
        s1_id = c_pair["s1_entity_id"].strip()
        t_id = c_pair["target_entity_id"].strip()
        rules = c_pair.get("rules_triggered", "")

        s1_rec = s1_recs.get(s1_id, {"entity_id": s1_id})
        t_rec = target_recs.get(t_id, {"entity_id": t_id})

        feat_dict = extract_pair_features(s1_rec, t_rec, rules_triggered=rules)

        if not is_test:
            # Ground truth label
            is_match = 1 if (t_id in gt_map.get(s1_id, set())) else 0
            is_val = 1 if (s1_id in val_ids) else 0
            feat_dict["is_match"] = is_match
            feat_dict["is_validation"] = is_val
            if is_match == 1:
                pos_count += 1

        feature_rows.append(feat_dict)

    logger.info(f"Feature extraction complete ({len(feature_rows):,} pairs, Positives: {pos_count:,}).")

    # 5. Save Parquet and TSV
    try:
        import pandas as pd
        df = pd.DataFrame(feature_rows)
        df.to_parquet(out_parquet, index=False)
        logger.info(f"Saved features to Parquet: {out_parquet}")
    except Exception:
        pass

    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(feature_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(feature_rows)

    manifest = {
        "num_pairs": len(feature_rows),
        "num_features": len(feature_rows[0]) - (4 if not is_test else 2),
        "positive_pairs": pos_count if not is_test else None,
        "parquet_path": str(out_parquet),
        "tsv_path": str(out_tsv),
    }

    with open(feat_dir / f"{out_stem}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
