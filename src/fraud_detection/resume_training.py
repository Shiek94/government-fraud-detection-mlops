# Resume training from a checkpoint. Make it possible to retrain the ML model on demand
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from fraud_detection.data_generation import generate_applications
from sklearn.base import clone
from sklearn.metrics import f1_score
import mlflow

# Construct project & checkpoint path
project_root = Path(__file__).resolve().parents[2]

checkpoint_dir = project_root / "models" / "checkpoints"
checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.joblib"))
sorted_checkpoints = sorted(checkpoint_files)

if not sorted_checkpoints:
    raise FileNotFoundError("No training checkpoints found!")

checkpoint_path = sorted_checkpoints[-1]
print(sorted_checkpoints)
print(checkpoint_path)

# Load the checkpoint
checkpoint = joblib.load(checkpoint_path)
checkpoint_df = checkpoint["cumulative_training_df"]
checkpoint_pipeline = checkpoint["current_pipeline"]
checkpoint_threshold = checkpoint["threshold"]
checkpoint_model_version = checkpoint["current_model_version"]

print(f"Training observations: {checkpoint_df.shape[0]}")
print(f"Current approved model: {checkpoint_model_version}")
print(f"Threshold: {checkpoint_threshold}")

# Increase training month logic, with yearly transition
checkpoint_month = checkpoint["month"]

year = int(checkpoint_month.split("-")[0])
month_number = int(checkpoint_month.split("-")[1])

if month_number == 12:
    month_number = 1
    year += 1
else:
    month_number += 1

next_training_month = f"{year}-{month_number:02d}"

if year > 2026:
    print("Simulation completed. No further retraining required.")
    raise SystemExit(0)

print(next_training_month)

# Generate applications
monthly_applications = generate_applications(n_samples=1000, seed=month_number, month=next_training_month, request_multiplier=1.5 if month_number >= 7 else 1.0)

# Split the data
X = monthly_applications[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
y = monthly_applications["fraud_confirmed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
assert set(X_train.index).isdisjoint(set(X_test.index))

# Recover training & evaluation df
checkpoint_training_df = monthly_applications.loc[X_train.index].copy()
checkpoint_eval_df = monthly_applications.loc[X_test.index].copy()

cumulative_checkpoint_df = pd.concat([checkpoint_df, checkpoint_training_df], ignore_index=True)

# Split the data
X_candidate = cumulative_checkpoint_df[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
y_candidate = cumulative_checkpoint_df["fraud_confirmed"]
X_eval = checkpoint_eval_df[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
y_eval = checkpoint_eval_df["fraud_confirmed"]

# Fit the candidate checkpoint pipeline
checkpoint_candidate_pipeline = clone(checkpoint_pipeline)
checkpoint_candidate_pipeline.fit(X_candidate, y_candidate)

# Generate fraud probabilities for both piples
current_pipeline_probabilities = checkpoint_pipeline.predict_proba(X_eval)[:, 1]
checkpoint_pipeline_probabilities = checkpoint_candidate_pipeline.predict_proba(X_eval)[:, 1]

current_pipeline_predictions = (current_pipeline_probabilities >= checkpoint_threshold).astype(int)
checkpoint_pipeline_prediction = (checkpoint_pipeline_probabilities >= checkpoint_threshold).astype(int)

different_predictions = (
    current_pipeline_predictions != checkpoint_pipeline_prediction
).sum()

print("Number of different predictions:", different_predictions)

import numpy as np

print(
    "Probabilities identical:",
    np.allclose(
        current_pipeline_probabilities,
        checkpoint_pipeline_probabilities
    )
)

current_f1 = f1_score(y_eval, current_pipeline_predictions)
candidate_f1 = f1_score(y_eval, checkpoint_pipeline_prediction)

# Model promotion logic
promotion = candidate_f1 > current_f1
if promotion:
    current_pipeline = checkpoint_candidate_pipeline
    current_model_version = f"candidate-{next_training_month}"
    print("Model promoted!")
else:
    current_pipeline = checkpoint_pipeline
    current_model_version = checkpoint_model_version
    print("Model not promoted")

# Create checkpoint dictionary & save it
checkpoint_dict = {
    "cumulative_training_df": cumulative_checkpoint_df,
    "current_pipeline": current_pipeline,
    "current_model_version": current_model_version,
    "threshold": checkpoint_threshold,
    "month": next_training_month
}

# Construct Checkpoint saving paths
next_checkpoint_path = project_root / "models" / "checkpoints" / f"checkpoint_{next_training_month}.joblib"
candidate_checkpoint_path = project_root / "models" / "candidates" / f"resume_candidate_{next_training_month}.joblib"

# Save and load checkpoint
next_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
candidate_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

# Save separate candidate artifact dictionary
candidate_artifact_dict = {
    "pipeline": checkpoint_candidate_pipeline,
    "threshold": checkpoint_threshold,
    "model_version": f"candidate-{next_training_month}"
}

if next_checkpoint_path.exists():
    raise FileExistsError(f"Checkpoint already exists: {next_checkpoint_path}")

if candidate_checkpoint_path.exists():
    raise FileExistsError(f"Checkpoint already exists: {candidate_checkpoint_path}")

joblib.dump(checkpoint_dict, next_checkpoint_path)
loaded_checkpoint_dict = joblib.load(next_checkpoint_path)
joblib.dump(candidate_artifact_dict, candidate_checkpoint_path)

# mlflow configuration
tracking_db_path = project_root / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path.as_posix()}")
mlflow.set_experiment("government-fraud-detection")

# Create a mlflow run
with mlflow.start_run(run_name=f"resume-{next_training_month}"):
    mlflow.log_param("Month", next_training_month)
    mlflow.log_param("Training_rows", cumulative_checkpoint_df.shape[0])
    mlflow.log_param("Threshold", checkpoint_threshold)
    mlflow.log_metric("Current F1", current_f1)
    mlflow.log_metric("Candidate F1", candidate_f1)
    mlflow.set_tag("Promotion decision", "Promoted" if promotion else "Denied")
    mlflow.set_tag("Previous_approved_version", checkpoint_model_version)
    mlflow.set_tag("Current_approved_version", current_model_version)
    mlflow.log_artifact(str(candidate_checkpoint_path), artifact_path="model")

# Update serving artifact if promoted
if promotion:
    approved_model_path = project_root / "models" / "fraud_model_approved.joblib"
    approved_artifact_dict = {
        "pipeline": current_pipeline,
        "threshold": checkpoint_threshold,
        "model_version": current_model_version
    }
    joblib.dump(approved_artifact_dict, approved_model_path)

print("Checkpoint month:", loaded_checkpoint_dict["month"])
print("Training rows:", loaded_checkpoint_dict["cumulative_training_df"].shape[0])
print("Approved model:", loaded_checkpoint_dict["current_model_version"])

print(f"Current F1: {current_f1}")
print(f"Candidate F1: {candidate_f1}")
print(f"Promote candidate: {promotion}")


print(cumulative_checkpoint_df.shape)
print(checkpoint_training_df.shape)
print(checkpoint_eval_df.shape)

