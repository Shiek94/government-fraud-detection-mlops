import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from data_generation import generate_applications
import joblib
from pathlib import Path
from sklearn.base import clone
import mlflow

# Generate training dataset
historical_df = generate_applications(n_samples=2000, seed=100, month="2025-12")

# Split df into features and labels
X = historical_df[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
y = historical_df["fraud_confirmed"]

# predeclare datatypes
X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.Series
y_test: pd.Series

# training and testing splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Recover the original historical training observations
historical_training_df = historical_df.loc[X_train.index].copy()

# Load the currently approved production model
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "fraud_model.joblib"
approved_model_path = project_root / "models" / "fraud_model_approved.joblib"

current_artifact = joblib.load(model_path)
current_pipeline = current_artifact["pipeline"]
threshold = current_artifact["threshold"]
current_model_version = current_artifact["model_version"]

# Generate monthly string and create dataframes for each month of 2026
cumulative_training_df = historical_training_df.copy()

# Configure mlflow
tracking_db_path = project_root / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path.as_posix()}")
mlflow.set_experiment("government-fraud-detection")

# Data drift detection reference
reference_requested_mean = historical_training_df["requested_amount"].mean()
drift_threshold = 0.25

for month_number in range(1, 13):
    month = f"2026-{month_number:02d}"

    # Data drift simulation
    request_multiplier = 1.5 if month_number >= 7 else 1.0
    monthly_df = generate_applications(n_samples=1000, seed=month_number, month=month, request_multiplier=request_multiplier)

    # Data drift detection logic
    monthly_requested_mean = monthly_df["requested_amount"].mean()
    relative_change = (abs(monthly_requested_mean - reference_requested_mean)) / reference_requested_mean
    drift_detected = relative_change >= drift_threshold

    # Split the generated data in training and evaluation
    X = monthly_df[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
    y = monthly_df["fraud_confirmed"]

    X_train_month, X_test_month, y_train_month, y_test_month = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    monthly_training_df = monthly_df.loc[X_train_month.index].copy()
    monthly_eval_df = monthly_df.loc[X_test_month.index].copy()

    # Update currently used training df
    cumulative_training_df = pd.concat([cumulative_training_df, monthly_training_df], ignore_index=True)

    # Split the data
    X_candidate = cumulative_training_df[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
    y_candidate = cumulative_training_df["fraud_confirmed"]
    X_eval = monthly_eval_df[["monthly_income", "requested_amount", "household_size", "previous_applications", "employment_status"]]
    y_eval = monthly_eval_df["fraud_confirmed"]

    # Clone the current pipeline into the candidate pipeline
    candidate_pipeline = clone(current_pipeline)
    candidate_pipeline.fit(X_candidate, y_candidate)

    # Obtain fraud probabilities from current and candidate pipelines
    current_probabilities = current_pipeline.predict_proba(X_eval)[:, 1]
    candidate_probabilities = candidate_pipeline.predict_proba(X_eval)[:, 1]

    # Convert to binary classifications
    predictions_current = (current_probabilities >= threshold).astype(int)
    predictions_candidate = (candidate_probabilities >= threshold).astype(int)

    # Calculate Metrics for both models
    accuracy_current = accuracy_score(y_eval, predictions_current)
    precision_current = precision_score(y_eval, predictions_current, zero_division=0)
    recall_current = recall_score(y_eval, predictions_current)
    f1_current = f1_score(y_eval, predictions_current)

    accuracy_candidate = accuracy_score(y_eval, predictions_candidate)
    precision_candidate = precision_score(y_eval, predictions_candidate, zero_division=0)
    recall_candidate = recall_score(y_eval, predictions_candidate)
    f1_candidate = f1_score(y_eval, predictions_candidate)

    # Create candidate artifact dictionary
    candidate_artifact = {
        "pipeline": candidate_pipeline,
        "threshold": threshold,
        "model_version": f"candidate-{month}"
    }

    # Model version & promotion tracking
    previous_approved_version = current_model_version
    candidate_model_version = candidate_artifact["model_version"]

    # Model promotion logic
    promote_candidate = f1_candidate > f1_current

    if promote_candidate:
        current_pipeline = candidate_pipeline
        promotion_decision = "Promoted"
        current_model_version = candidate_model_version
    else:
        promotion_decision = "Denied"

    # Save a checkpoint of the current simulation state
    if month_number == 6:
        checkpoint = {
            "cumulative_training_df": cumulative_training_df,
            "current_pipeline": current_pipeline,
            "current_model_version": current_model_version,
            "threshold": threshold,
            "month": month
        }

        checkpoint_path = project_root / "models" / "checkpoints" / "checkpoint_2026-06.joblib"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(checkpoint, checkpoint_path)
        loaded_checkpoint_dict = joblib.load(checkpoint_path)
        number_training_observations = loaded_checkpoint_dict["cumulative_training_df"].shape[0]
        model_version = loaded_checkpoint_dict["current_model_version"]
        print(number_training_observations)
        print(model_version)

    # Construct dynamic path for saving model artifacts
    candidate_path = project_root / "models" / "candidates" / f"fraud_candidate_{month}.joblib"

    # Save artifact dict to path
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_artifact, candidate_path)

    # mlflow logging
    with mlflow.start_run(run_name=f"retraining-{month}"):
        mlflow.log_param("Month", month)
        mlflow.log_param("Training_rows", cumulative_training_df.shape[0])
        mlflow.log_param("Threshold", threshold)
        mlflow.log_param("Drift_threshold", drift_threshold)
        mlflow.log_param("Reference_requested_mean", reference_requested_mean)
        mlflow.log_metric("Current Accuracy", accuracy_current)
        mlflow.log_metric("Current Precision", precision_current)
        mlflow.log_metric("Current Recall", recall_current)
        mlflow.log_metric("Current F1", f1_current)
        mlflow.log_metric("Candidate Accuracy", accuracy_candidate)
        mlflow.log_metric("Candidate Precision", precision_candidate)
        mlflow.log_metric("Candidate Recall", recall_candidate)
        mlflow.log_metric("Candidate F1", f1_candidate)
        mlflow.log_metric("Average requested amount", monthly_requested_mean)
        mlflow.log_metric("Relative requested amount change", relative_change)
        mlflow.set_tag("Requested amount drift detected", str(drift_detected))
        mlflow.set_tag("Promotion decision", promotion_decision)
        mlflow.set_tag("Previous_approved_version", previous_approved_version)
        mlflow.set_tag("Candidate_version", candidate_model_version)
        mlflow.set_tag("Current_approved_version", current_model_version)
        mlflow.log_artifact(str(candidate_path), artifact_path="model")

    print(f"{month} | " f"Relative change: {relative_change:.2%} | " f"Drift detected: {drift_detected}")
    print(f"{month} | Mean requested amount: {monthly_df['requested_amount'].mean():.2f}")
    print(f"{month} | Training: {cumulative_training_df.shape[0]} | Current F1: {f1_current} | Candidate F1: {f1_candidate} Promoted: {promote_candidate}")
    print(monthly_df["fraud_confirmed"].mean())

# Create approved_artifact dictionary
approved_artifact = {
    "pipeline": current_pipeline,
    "threshold": threshold,
    "model_version": current_model_version
}

# Save the approved model dict
joblib.dump(approved_artifact, approved_model_path)
loaded_approved_model_dict = joblib.load(approved_model_path)
if loaded_approved_model_dict["model_version"] == current_model_version:
    print("Model versions match!")
    print(f"Model version: {loaded_approved_model_dict['model_version']}")
else:
    print("Model version do not match!")



