import os
import json
from pathlib import Path
from src.training.train import load_dataset, extract_matrix

def test_load_dataset_and_extract_matrix(tmp_path):
    tsv_file = tmp_path / "train_features.tsv"
    
    with open(tsv_file, "w") as f:
        f.write("s1_entity_id\ttarget_entity_id\tis_match\tis_validation\tfeat1\tfeat2\n")
        f.write("s1\tt1\t1\t0\t0.5\t1.0\n")
        f.write("s1\tt2\t0\t0\t0.1\t0.2\n")
        f.write("s2\tt3\t1\t1\t0.9\t0.9\n")
        f.write("s2\tt4\t0\t1\t0.2\t0.3\n")
        
    train_rows, val_rows, feature_names = load_dataset(tsv_file)
    
    assert len(train_rows) == 2
    assert len(val_rows) == 2
    assert feature_names == ["feat1", "feat2"]
    
    X_train, y_train = extract_matrix(train_rows, feature_names)
    assert len(X_train) == 2
    assert X_train[0] == [0.5, 1.0]
    assert y_train == [1, 0]
    
    X_val, y_val = extract_matrix(val_rows, feature_names)
    assert len(X_val) == 2
    assert X_val[1] == [0.2, 0.3]
    assert y_val == [1, 0]
