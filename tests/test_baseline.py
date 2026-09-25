import json
from pathlib import Path
from src.utils.storage import StorageManager
from src.models.baseline import run_baseline

def test_run_baseline(tmp_path):
    """Test the rule-based baseline model execution."""
    # 1. Setup mock environment
    config = {
        "pipeline": {"beta": 0.5},
        "dataset": {"files": {"train_ground_truth": "train_ground_truth.tsv"}},
        "models": {
            "baseline": {
                "name_weight": 0.5,
                "address_weight": 0.5,
                "country_weight": 0.0,
                "default_threshold": 0.6
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
    storage.experiments_dir = storage.storage_root / "experiments"
    storage.initialize_directories()

    # 2. Create mock validation split
    with open(storage.splits_dir / "validation_split.json", "w") as f:
        json.dump({"val_entity_ids": ["s1", "s2"]}, f)

    # 3. Create mock ground truth
    with open(storage.raw_dir / "train_ground_truth.tsv", "w") as f:
        f.write("s1_entity_id\ttarget_entity_ids\n")
        f.write("s1\tt1,t2\n")
        f.write("s2\tt3\n")

    # 4. Create mock features TSV
    with open(storage.features_dir / "train_features.tsv", "w") as f:
        f.write("s1_entity_id\ttarget_entity_id\tis_validation\tname_token_set_ratio\taddr_token_set_ratio\tcountry_exact_match\n")
        # Match for s1->t1: 100 name, 100 addr => score 1.0 (>=0.6) => Match
        f.write("s1\tt1\t1\t100\t100\t1\n")
        # Miss for s1->t2 (FN): 0 name, 0 addr => score 0.0 => Miss
        f.write("s1\tt2\t1\t0\t0\t0\n")
        # Match for s2->t3: 50 name, 100 addr => score 0.75 (>=0.6) => Match
        f.write("s2\tt3\t1\t50\t100\t1\n")

    # 5. Run baseline
    report = run_baseline(config, storage)

    # 6. Validate
    assert "metrics" in report
    metrics = report["metrics"]
    
    # Ground Truth:
    # s1 -> {t1, t2}
    # s2 -> {t3}
    # Predictions:
    # s1 -> {t1}
    # s2 -> {t3}
    
    # s1: tp=1, fp=0, fn=1 => prec=1.0, rec=0.5
    # f0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    
    # s2: tp=1, fp=0, fn=0 => prec=1.0, rec=1.0, f0.5=1.0
    
    # macro_f0.5 = (0.8333 + 1.0) / 2 = 0.9167
    
    assert abs(metrics["macro_f0.5"] - 0.916666) < 1e-4
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 0.75

