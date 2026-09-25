# Amazon ML Challenge — Business Entity Resolution

A modular, reproducible, production-grade Machine Learning pipeline for large-scale multi-source Business Entity Resolution.

---

## Architecture Overview

```
Local / Antigravity                GitHub                     Kaggle / Compute
  [Code & Config]  ──git push──>  [Master Repo]  ──git pull──>  [Execution Runtime]
                                                                        │
                                                                        ▼
                                                                Google Drive (Persistent)
                                                                ├── Raw 1GB Dataset (.zip)
                                                                ├── Checkpoints (latest / best)
                                                                ├── Experiments & Logs
                                                                └── Final Submissions
```

---

## Directory Structure

```
amazon-ml-entity-resolution/
├── README.md                      # Project documentation and quickstart
├── requirements.txt               # Pinned Python dependencies
├── .gitignore                     # Excludes raw data, checkpoints, and secrets
├── configs/
│   ├── config.yaml                # Master pipeline configuration
│   └── experiments/               # Experiment-specific configurations
├── src/
│   ├── data/                      # Data loading, audit, and EDA
│   ├── preprocessing/             # Text normalization and cleansing
│   ├── blocking/                  # Multi-strategy candidate generation
│   ├── features/                  # Pairwise similarity & cross-field features
│   ├── models/                    # Baseline and ML matching models
│   ├── training/                  # Training loops, checkpointing, early stopping
│   ├── inference/                 # Test set prediction pipeline
│   ├── evaluation/                # Macro F0.5 metrics & validation harness
│   ├── submission/                # Validator and submission packager
│   └── utils/                     # Storage, checkpoints, and stage runner
├── notebooks/
│   ├── 00_master_orchestrator.ipynb # Master Kaggle execution notebook
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_validation.ipynb
│   ├── 04_blocking.ipynb
│   ├── 05_features.ipynb
│   ├── 06_baseline.ipynb
│   ├── 07_training.ipynb
│   ├── 08_evaluation.ipynb
│   └── 09_inference.ipynb
├── scripts/
│   ├── run_stage.py               # Master stage CLI runner
│   ├── setup_kaggle.py            # Kaggle environment verification
│   ├── preprocess.py
│   ├── generate_candidates.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate_submission.py
├── docs/
│   ├── methodology.md             # Technical methodology documentation
│   └── storage_and_compute.md     # Storage & Kaggle workflow instructions
└── tests/
    └── test_metrics.py            # Metric and validation unit tests
```

---

## Stage Control System

Each stage can be executed independently. Cached artifacts are reused automatically without redundant recomputation:

```bash
# Run any stage via CLI
python scripts/run_stage.py --stage data_audit
python scripts/run_stage.py --stage preprocessing
python scripts/run_stage.py --stage blocking
python scripts/run_stage.py --stage features
python scripts/run_stage.py --stage training
python scripts/run_stage.py --stage threshold_optimization
python scripts/run_stage.py --stage test_inference
python scripts/run_stage.py --stage submission

# Force recomputation of an existing stage
python scripts/run_stage.py --stage preprocessing --force
```

---

## Evaluation Metric

The competition optimizes **Macro-Averaged $F_{0.5}$** across Source 1 entities:

$$F_{0.5} = (1 + 0.5^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = 1.25 \cdot \frac{P \cdot R}{0.25 \cdot P + R}$$

* **Precision-heavy**: False positive merges are heavily penalized.
* **Singletons**: True singletons correctly predicted with no matches receive full score (1.0). Inappropriately matched singletons score 0.0.
