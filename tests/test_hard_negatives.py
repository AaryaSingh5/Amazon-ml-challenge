import json
import csv
from pathlib import Path
from src.utils.storage import StorageManager
from src.features.hard_negatives import run_hard_negatives


class MockModel:
    """Predicts high confidence for pair s1->t2 (which is a non-match -> hard negative)."""
    def predict(self, X):
        # 4 rows: s1->t1 (match, pred high), s1->t2 (non-match, pred high), s2->t3 (match, pred low), s2->t4 (non-match, pred low)
        return [0.9, 0.8, 0.4, 0.1]


def test_hard_negatives(tmp_path):
    import joblib

    config = {
        "pipeline": {"beta": 0.5},
        "dataset": {"files": {"train_ground_truth": "train_ground_truth.tsv"}, "separator": "\t", "encoding": "utf-8"},
    }

    storage = StorageManager("configs/config.yaml")
    storage.data_root = tmp_path / "data"
    storage.storage_root = tmp_path / "artifacts"
    storage.raw_dir = storage.data_root / "raw"
    storage.splits_dir = storage.data_root / "splits"
    storage.features_dir = storage.data_root / "features"
    storage.checkpoints_dir = storage.storage_root / "checkpoints"
    storage.artifacts_dir = storage.storage_root / "artifacts_out"
    storage.initialize_directories()

    # Save mock model
    model_dir = storage.checkpoints_dir / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(MockModel(), model_dir / "model.joblib")

    # Mock ground truth (s1->t1 match, s2->t3 match; s1->t2 and s2->t4 are non-matches)
    with open(storage.raw_dir / "train_ground_truth.tsv", "w") as f:
        f.write("s1_entity_id\ttarget_entity_ids\n")
        f.write("s1\tt1\n")
        f.write("s2\tt3\n")

    # Mock features: 4 pairs, 2 in train, 2 in val
    with open(storage.features_dir / "train_features.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["s1_entity_id", "target_entity_id", "is_match", "is_validation", "feat1"])
        writer.writerow(["s1", "t1", "1", "0", "0.9"])  # TP: match, pred high
        writer.writerow(["s1", "t2", "0", "0", "0.8"])  # FP: non-match, pred high -> hard neg
        writer.writerow(["s2", "t3", "1", "1", "0.4"])  # FN: match, pred low
        writer.writerow(["s2", "t4", "0", "1", "0.1"])  # TN: non-match, pred low

    report = run_hard_negatives(config, storage)

    assert report["stats"]["true_positives"] == 1
    assert report["stats"]["false_positives"] == 1
    assert report["stats"]["false_negatives"] == 1
    assert report["stats"]["true_negatives"] == 1
    assert report["stats"]["hard_negatives_mined"] == 1  # s1->t2 is the hard neg (pred=0.8 >= 0.5)

    # Verify mined tsv has original 4 rows + 1 hard neg appended = 5 rows total
    mined_tsv = storage.features_dir / "train_features_mined.tsv"
    assert mined_tsv.exists()
    with open(mined_tsv, "r") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 5, f"Expected 5 rows but got {len(rows)}"
    # Last row is the hard negative s1->t2
    assert rows[-1]["s1_entity_id"] == "s1"
    assert rows[-1]["target_entity_id"] == "t2"
    assert rows[-1]["is_hard_negative"] == "1"
    print("All assertions passed.")


if __name__ == "__main__":
    from pathlib import Path
    import shutil
    tmp = Path("tmp_test_hn")
    tmp.mkdir(exist_ok=True)
    try:
        test_hard_negatives(tmp)
        print("Test Passed!")
    finally:
        shutil.rmtree(tmp)
