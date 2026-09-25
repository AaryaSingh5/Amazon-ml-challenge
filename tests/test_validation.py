"""Unit tests for Stage 4: Validation Split and Evaluation module."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.split import create_entity_validation_split


def test_entity_validation_split_no_leakage_and_coverage():
    s1_entities = [{"entity_id": f"S1_{i}", "country": "US" if i % 2 == 0 else "INDIA"} for i in range(100)]
    gt_map = {
        f"S1_{i}": [f"S2_{i*10}"] if i < 60 else ([f"S2_{i*10}", f"S3_{i*10}"] if i < 80 else [])
        for i in range(100)
    }

    split = create_entity_validation_split(
        s1_entities=s1_entities,
        ground_truth_map=gt_map,
        val_ratio=0.20,
        random_seed=42,
    )

    train_ids = set(split["train_entity_ids"])
    val_ids = set(split["val_entity_ids"])

    # 1. Zero leakage guarantee
    assert len(train_ids.intersection(val_ids)) == 0

    # 2. 100% Entity coverage
    assert len(train_ids) + len(val_ids) == 100
    assert len(val_ids) == 20
    assert len(train_ids) == 80

    # 3. Stratification balance
    t_stats = split["train_statistics"]
    v_stats = split["validation_statistics"]
    # Total singletons in data = 20 (20%), so val should have ~4 singletons (20%)
    assert abs(t_stats["singletons_pct"] - v_stats["singletons_pct"]) < 5.0
    assert abs(t_stats["one_to_many_pct"] - v_stats["one_to_many_pct"]) < 5.0


def test_split_reproducibility():
    s1_entities = [{"entity_id": f"S1_{i}", "country": "US"} for i in range(50)]
    gt_map = {f"S1_{i}": [f"S2_{i}"] for i in range(30)}

    split1 = create_entity_validation_split(s1_entities, gt_map, val_ratio=0.20, random_seed=42)
    split2 = create_entity_validation_split(s1_entities, gt_map, val_ratio=0.20, random_seed=42)

    assert split1["train_entity_ids"] == split2["train_entity_ids"]
    assert split1["val_entity_ids"] == split2["val_entity_ids"]


if __name__ == "__main__":
    test_entity_validation_split_no_leakage_and_coverage()
    test_split_reproducibility()
    print("All Stage 4 Validation unit tests passed successfully!")
