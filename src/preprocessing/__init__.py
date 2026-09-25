"""Preprocessing package init."""
from __future__ import annotations

from src.preprocessing.normalizer import (
    normalize_business_name,
    strip_legal_suffix,
    normalize_address,
    extract_postal_code,
    extract_address_digits,
    normalize_country,
    transform_record,
)
from src.preprocessing.pipeline import run_preprocessing, process_table_file

__all__ = [
    "normalize_business_name",
    "strip_legal_suffix",
    "normalize_address",
    "extract_postal_code",
    "extract_address_digits",
    "normalize_country",
    "transform_record",
    "run_preprocessing",
    "process_table_file",
]
