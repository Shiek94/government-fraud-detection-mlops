from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
from pathlib import Path

class ApplicationInput(BaseModel):
    monthly_income: int = Field(ge=0)
    requested_amount: int = Field(gt=0)
    household_size: int = Field(ge=1)
    previous_applications: int = Field(ge=0)
    employment_status: Literal["employed", "unemployed", "self-employed"]

# Build path to saved model & load it
project_root = Path(__file__).resolve().parents[2]
model_path = project_root/"models"/"fraud_model.joblib"

artifact = joblib.load(model_path)
model = artifact["pipeline"]
threshold = artifact["threshold"]
model_version = artifact["model_version"]

print(f'Model version: {model_version}')
print(f'Classification threshold: {threshold}')

app = FastAPI()

@app.get("/health")
def api_health_check():
    return {"status": "OK"}

@app.post("/validate")
def application_validation(application: ApplicationInput):
    application_to_dict = application.model_dump()
    application_to_df = pd.DataFrame([application_to_dict])

    return {
        "valid": True,
        "monthly_income": application.monthly_income,
        "employment_status": application.employment_status,
    }

@app.post("/predict")
def predict_fraud(application: ApplicationInput):
    application_to_dict = application.model_dump()
    application_to_df = pd.DataFrame([application_to_dict])

    fraud_probability = float(model.predict_proba(application_to_df)[0, 1])
    flagged_for_review = bool(fraud_probability >= threshold)

    return {
        "fraud_probability": fraud_probability,
        "flagged_for_review": flagged_for_review,
        "model_version": model_version
    }



