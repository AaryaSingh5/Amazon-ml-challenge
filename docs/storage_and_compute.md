# Storage & Compute Architecture

## 1. Master Dataset Location
* **Authoritative Location**: Google Drive
* **Master Archive**: `drive/MyDrive/Project file/6ab10eb3b23ba_student_resource.zip`
* **Rule**: Raw ~1 GB archive is never uploaded to GitHub.

## 2. Checkpoints & Durable Artifacts
* Checkpoints are stored in persistent storage:
  - `checkpoints/latest/` (most recent training state + model.joblib)
  - `checkpoints/best/` (highest macro F0.5 model)
  - `checkpoints/experiments/<exp_id>/` (experiment archive)
* Every checkpoint contains:
  - `model.joblib`
  - `metadata.json` (git commit hash, F0.5 score, hyperparameters, random seed, timestamp)
  - Preprocessor/scaler artifacts

## 3. Kaggle Compute Workflow
1. Kaggle notebook clones or pulls repository from GitHub (`git pull origin main`).
2. Google Drive is mounted or dataset is synchronized locally.
3. Individual stages are executed independently via `src.utils.stage_runner.StageController` or `scripts/run_stage.py`.
4. Processed artifacts and checkpoints are saved back to durable storage.
