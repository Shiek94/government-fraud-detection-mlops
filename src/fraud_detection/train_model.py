from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from data_generation import generate_applications
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)


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

# split into training and validation sets
X_train_small, X_validation, y_train_small, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

# one-hot-encode categorical feature employment_status
preprocessor = ColumnTransformer(
    transformers=[
        ("employment_status_encoder", OneHotEncoder(handle_unknown="ignore"), ["employment_status"])
    ],
    remainder="passthrough"
)

# create the pipeline
pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("Classifier", DecisionTreeClassifier(max_depth=3, random_state=42))
])




if __name__ == "__main__":
    # apply the pipeline
    pipeline.fit(X_train_small, y_train_small)
    y_prob = pipeline.predict_proba(X_validation)[:, 1]
    print(y_prob[:10])
    print("Maximum predicted probability:", y_prob.max())

    # scout for the best performing classification threshold percentage
    """for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
        y_pred = (y_prob >= threshold).astype(int)

        # evaluate metrics
        accuracy = accuracy_score(y_validation, y_pred)
        precision = precision_score(y_validation, y_pred, zero_division=0)
        recall = recall_score(y_validation, y_pred)
        f1 = f1_score(y_validation, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_validation, y_pred)

        print(f"Metrics for threshold: {threshold}")
        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1 score: {f1}")
        print(f"confusion matrix:{conf_matrix}")"""

    # chose 0.25 as best performing classification threshold
    threshold = 0.25

    # fit the pipeline on the whole training set
    pipeline.fit(X_train, y_train)

    # calculate binary labels from predicted probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Metrics for threshold: {threshold}")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1 score: {f1}")
    print(f"confusion matrix:{conf_matrix}")
