"""Unit tests for Stage 1: Data Audit module."""

import sys
import tempfile
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import audit_table_file, audit_ground_truth_file, run_data_audit, compute_text_stats
from src.utils.storage import StorageManager


def test_compute_text_stats():
    values = ["Alpha Corp", "Beta LLC", "Gamma Inc", "", None, "Delta LLC"]
    stats = compute_text_stats(values)

    assert stats["count"] == 4
    assert stats["missing"] == 2
    assert stats["unique_count"] == 4
    assert stats["char_len_min"] == 8
    assert stats["char_len_max"] == 10
    assert stats["token_count_mean"] == 2.0


def test_audit_table_file_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        tpath = Path(tmpdir) / "test_table.tsv"
        with open(tpath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S1_1", "Acme Corp", "123 Main St", "US"])
            writer.writerow(["S1_2", "Beta LLC", "456 Oak Ave", "India"])
            writer.writerow(["S1_3", "Gamma Inc", "789 Pine Rd", "US"])
            writer.writerow(["S1_3", "Gamma Inc", "789 Pine Rd", "US"])  # Duplicate row

        audit = audit_table_file(tpath, "test_table")

        assert audit["num_rows"] == 4
        assert audit["unique_entity_ids"] == 3
        assert audit["duplicate_entity_ids"] == 1
        assert audit["total_exact_duplicate_rows"] == 1
        assert audit["country_distribution"]["US"] == 3
        assert audit["country_distribution"]["India"] == 1


def test_audit_ground_truth_cardinalities():
    with tempfile.TemporaryDirectory() as tmpdir:
        s1_path = Path(tmpdir) / "s1.tsv"
        gt_path = Path(tmpdir) / "gt.tsv"

        with open(s1_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "name", "address", "country"])
            writer.writerow(["S1_1", "A", "Addr1", "US"])
            writer.writerow(["S1_2", "B", "Addr2", "US"])
            writer.writerow(["S1_3", "C", "Addr3", "US"])
            writer.writerow(["S1_4", "D", "Addr4", "US"])

        with open(gt_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "match_id"])
            writer.writerow(["S1_1", "S2_100"])
            writer.writerow(["S1_3", "S2_200,S3_300"])

        audit = audit_ground_truth_file(gt_path, s1_path)

        assert audit["total_source1_entities"] == 4
        # S1_2 and S1_4 have 0 matches -> 2 singletons
        assert audit["singletons_count"] == 2
        assert audit["singletons_pct"] == 50.0
        # S1_1 has 1 match
        assert audit["one_to_one_count"] == 1
        # S1_3 has 2 matches
        assert audit["one_to_many_count"] == 1
        assert audit["max_matches_per_s1"] == 2
        assert audit["matched_targets_source2"] == 2
        assert audit["matched_targets_source3"] == 1


if __name__ == "__main__":
    test_compute_text_stats()
    test_audit_table_file_metrics()
    test_audit_ground_truth_cardinalities()
    print("All Stage 1 audit unit tests passed successfully!")
