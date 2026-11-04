from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

class ApplicationInput(BaseModel):
    monthly_income: int = Field(ge=0)
    requested_amount: int = Field(gt=0)
    household_size: int = Field(ge=1)
    previous_applications: int = Field(ge=0)
    employment_status: Literal["employed", "unemployed", "self-employed"]

app = FastAPI()

@app.get("/health")
def api_health_check():
    return {"status": "OK"}

@app.post("/validate")
def application_validation(application: ApplicationInput):
    application_to_dict = application.model_dump()
    application_to_df = pd.DataFrame([application_to_dict])

    print(application_to_df)
    print(application_to_df.shape)

    return {
        "valid": True,
        "monthly_income": application.monthly_income,
        "employment_status": application.employment_status,
    }





