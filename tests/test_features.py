"""Unit tests for Stage 7: Feature Engineering module."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.similarity import (
    levenshtein_ratio,
    token_sort_ratio,
    token_set_ratio,
    jaccard_similarity,
    char_ngram_jaccard,
    numeric_tokens_jaccard,
    postal_code_agreement,
)
from src.features.pipeline import extract_pair_features
from src.preprocessing.normalizer import transform_record


def test_similarity_metrics():
    # Token sort invariance
    assert token_sort_ratio("apollo hospital", "hospital apollo") == 1.0
    # Token set handles subsets
    assert token_set_ratio("starbucks coffee", "starbucks coffee & tea inc") >= 0.90
    # Jaccard
    assert jaccard_similarity("a b c", "a b d") == 2.0 / 4.0
    # Numeric tokens
    assert numeric_tokens_jaccard("123 main st", "123 main st ste 4") >= 0.50
    # Postal agreement
    assert postal_code_agreement("10001", "10001") == 1.0
    assert postal_code_agreement("10001", "90210") == 0.0
    assert postal_code_agreement("10001", "") == 0.5


def test_extract_pair_features_structure():
    s1 = transform_record("S1_1", "Walmart Supercenter", "702 SW 8th St", "US")
    t1 = transform_record("S2_1", "Walmart", "702 Southwest 8th Street", "US")

    feats = extract_pair_features(s1, t1, rules_triggered="rule_exact_name,rule_exact_addr")

    assert feats["s1_entity_id"] == "S1_1"
    assert feats["target_entity_id"] == "S2_1"
    assert feats["name_token_set_ratio"] == 1.0
    assert feats["addr_token_set_ratio"] == 1.0
    assert feats["country_norm_match"] == 1.0
    assert feats["strong_agreement_indicator"] == 1.0
    assert feats["is_source2"] == 1.0
    assert feats["rule_exact_name_flag"] == 1.0


if __name__ == "__main__":
    test_similarity_metrics()
    test_extract_pair_features_structure()
    print("All Stage 7 Feature Engineering unit tests passed successfully!")
