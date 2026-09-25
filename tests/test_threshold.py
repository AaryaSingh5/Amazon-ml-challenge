import json
from pathlib import Path
from src.utils.storage import StorageManager
from src.evaluation.threshold_optimizer import run_threshold_optimization

class MockModel:
    def predict(self, X):
        return [0.1, 0.9, 0.4, 0.8]

def test_run_threshold_opt(tmp_path):
    import joblib
    
    # 1. Setup mock environment
    config = {
        "pipeline": {"beta": 0.5},
        "dataset": {"files": {"train_ground_truth": "train_ground_truth.tsv"}},
        "evaluation": {
            "threshold_grid_search": {
                "start": 0.20,
                "end": 0.95,
                "step": 0.10
            }
        },
        "storage": {
            "data_root": str(tmp_path / "data"),
            "storage_root": str(tmp_path / "artifacts"),
            "raw_dir": str(tmp_path / "data/raw"),
            "splits_dir": str(tmp_path / "data/splits"),
            "features_dir": str(tmp_path / "data/features"),
        }
    }
    
    storage = StorageManager("configs/config.yaml") # mock
    storage.data_root = tmp_path / "data"
    storage.storage_root = tmp_path / "artifacts"
    storage.raw_dir = storage.data_root / "raw"
    storage.splits_dir = storage.data_root / "splits"
    storage.features_dir = storage.data_root / "features"
    storage.checkpoints_dir = storage.storage_root / "checkpoints"
    storage.artifacts_dir = storage.storage_root / "artifacts"
    storage.initialize_directories()
    
    # Save mock model
    model_dir = storage.checkpoints_dir / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(MockModel(), model_dir / "model.joblib")

    # Mock validation split
    with open(storage.splits_dir / "validation_split.json", "w") as f:
        json.dump({"val_entity_ids": ["s1", "s2"]}, f)

    # Mock ground truth
    with open(storage.raw_dir / "train_ground_truth.tsv", "w") as f:
        f.write("s1_entity_id\ttarget_entity_ids\n")
        f.write("s1\tt1,t2\n")
        f.write("s2\tt3\n")

    # Mock features TSV
    with open(storage.features_dir / "train_features.tsv", "w") as f:
        f.write("s1_entity_id\ttarget_entity_id\tis_validation\tfeat1\n")
        # predictions will be 0.1, 0.9, 0.4, 0.8
        f.write("s1\tt1\t1\t1\n") # p=0.1
        f.write("s1\tt2\t1\t2\n") # p=0.9
        f.write("s2\tt3\t1\t3\n") # p=0.4
        f.write("s2\tt4\t1\t4\n") # p=0.8

    report = run_threshold_optimization(config, storage)
    
    assert "optimal_threshold" in report
    
    grid = report["grid_search_results"]
    assert len(grid) == 8 # 0.20 to 0.90 in steps of 0.10 (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

