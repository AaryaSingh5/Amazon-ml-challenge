"""Unit tests for Stage 3: Normalization and Preprocessing module."""

import sys
import tempfile
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalizer import (
    normalize_business_name,
    strip_legal_suffix,
    normalize_address,
    extract_postal_code,
    extract_address_digits,
    normalize_country,
    transform_record,
)
from src.preprocessing.pipeline import process_table_file


def test_business_name_normalization():
    assert normalize_business_name("Café Coffee Day & Co.") == "cafe coffee day and co"
    assert normalize_business_name("AT&T Inc.") == "at and t inc"
    assert normalize_business_name("McDonald's - Fast Food") == "mcdonald s fast food"


def test_legal_suffix_stripping():
    # Suffixes should be cleanly stripped from trailing position
    assert strip_legal_suffix("apple inc") == "apple"
    assert strip_legal_suffix("tata consultancy services pvt ltd") == "tata consultancy services"
    assert strip_legal_suffix("reliance industries limited") == "reliance industries"
    assert strip_legal_suffix("baker hughes a ge company") == "baker hughes a ge"
    # Non-suffix or single token should be preserved
    assert strip_legal_suffix("company") == "company"


def test_address_normalization_and_abbreviations():
    addr = "123 N. Main St., Ste. 400, New York, NY 10001"
    norm = normalize_address(addr)
    assert "north" in norm
    assert "main" in norm
    assert "street" in norm
    assert "suite" in norm
    assert "400" in norm


def test_postal_code_extraction():
    assert extract_postal_code("123 Main St, New York, NY 10001-1234") == "10001"
    assert extract_postal_code("MG Road, Bangalore, Karnataka 560001") == "560001"
    assert extract_postal_code("No Zip Here") == ""


def test_country_normalization_open_set():
    assert normalize_country("usa") == "US"
    assert normalize_country("United States of America") == "US"
    assert normalize_country("in") == "INDIA"
    assert normalize_country("India") == "INDIA"
    # Unseen country test (France, Germany, etc.)
    assert normalize_country("france") == "FRANCE"
    assert normalize_country("FR") == "FRANCE"
    assert normalize_country("Australia") == "AUSTRALIA"


def test_transform_record_preserves_originals():
    rec = transform_record(
        entity_id="S1_101",
        business_name="Café Amazon LLC",
        business_address="100 5th Ave, Fl 2",
        country="US",
    )
    assert rec["entity_id"] == "S1_101"
    assert rec["original_name"] == "Café Amazon LLC"
    assert rec["normalized_name"] == "cafe amazon llc"
    assert rec["clean_name_no_suffix"] == "cafe amazon"
    assert rec["original_address"] == "100 5th Ave, Fl 2"
    assert "avenue" in rec["normalized_address"]
    assert "floor" in rec["normalized_address"]
    assert rec["original_country"] == "US"
    assert rec["normalized_country"] == "US"


def test_process_table_file_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_tsv = tmp_path / "raw_s1.tsv"
        out_parquet = tmp_path / "s1_clean.parquet"

        with open(in_tsv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["entity_id", "business_name", "business_address", "country"])
            w.writerow(["S1_1", "Walmart Inc", "702 SW 8th St", "usa"])

        res = process_table_file(in_tsv, out_parquet)
        assert res["status"] == "success"
        assert res["num_records"] == 1
        assert Path(res["tsv_path"]).exists()


if __name__ == "__main__":
    test_business_name_normalization()
    test_legal_suffix_stripping()
    test_address_normalization_and_abbreviations()
    test_postal_code_extraction()
    test_country_normalization_open_set()
    test_transform_record_preserves_originals()
    test_process_table_file_end_to_end()
    print("All Stage 3 Preprocessing unit tests passed successfully!")
