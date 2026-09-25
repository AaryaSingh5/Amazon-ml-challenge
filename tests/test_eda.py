"""Unit tests for Stage 2: Exploratory Data Analysis (EDA) module."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda import (
    analyze_legal_suffixes,
    analyze_casing_and_punctuation,
    analyze_addresses,
    analyze_frequent_tokens,
)


def test_analyze_legal_suffixes():
    names = [
        "Apple Inc",
        "Alphabet LLC",
        "Tata Consultancy Services Ltd",
        "Reliance Industries Private Limited",
        "Unknown Shop",
    ]
    res = analyze_legal_suffixes(names)
    assert res["names_with_legal_suffix_count"] == 4
    assert res["names_with_legal_suffix_pct"] == 80.0
    assert "inc" in res["top_legal_suffixes"]
    assert "llc" in res["top_legal_suffixes"]


def test_analyze_casing_and_punctuation():
    texts = [
        "ACME CORP.",
        "starbucks coffee & tea",
        "McDonald's Restaurant",
        "Walmart Supercenter",
    ]
    res = analyze_casing_and_punctuation(texts)
    assert res["casing_distribution"]["all_uppercase"] == 1
    assert res["casing_distribution"]["all_lowercase"] == 1
    assert res["casing_distribution"]["title_case"] == 1
    assert "." in res["top_punctuation_marks"]
    assert "&" in res["top_punctuation_marks"]


def test_analyze_addresses():
    addrs = [
        "123 Main St, New York, NY 10001",
        "MG Road, Bangalore, Karnataka 560001",
        "Just A Plain Name Without Numbers",
    ]
    res = analyze_addresses(addrs)
    assert res["addresses_with_numbers_pct"] == 66.67
    assert res["addresses_with_5digit_us_postal_pct"] == 33.33
    assert res["addresses_with_6digit_india_pin_pct"] == 33.33
    assert "st" in res["top_address_abbreviations"]


def test_analyze_frequent_tokens():
    texts = ["Apollo Hospital", "Hospital Care", "Apollo Pharmacy"]
    toks = analyze_frequent_tokens(texts)
    assert toks["apollo"] == 2
    assert toks["hospital"] == 2


if __name__ == "__main__":
    test_analyze_legal_suffixes()
    test_analyze_casing_and_punctuation()
    test_analyze_addresses()
    test_analyze_frequent_tokens()
    print("All Stage 2 EDA unit tests passed successfully!")
