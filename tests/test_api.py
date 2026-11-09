from fastapi.testclient import TestClient
from fraud_detection.api import app

# Test API health
def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_predict_valid_application():
    application = {
        "monthly_income": 300,
        "requested_amount": 1200,
        "household_size": 2,
        "previous_applications": 0,
        "employment_status": "employed"
    }

    client = TestClient(app)
    response = client.post("/predict", json=application)

    assert response.status_code == 200
    result = response.json()

    assert 0 <= result["fraud_probability"] <= 1
    assert isinstance(result["flagged_for_review"], bool)
    assert isinstance(result["model_version"], str)
    assert len(result["model_version"]) > 0

def test_predict_invalid_application():
    application = {
        "monthly_income": -500,
        "requested_amount": -1200,
        "household_size": 2,
        "previous_applications": 0,
        "employment_status": "employed"
    }

    client = TestClient(app)
    response = client.post("/predict", json=application)
    result = response.json()

    error_locations = [error["loc"] for error in result["detail"]]

    assert ["body", "monthly_income"] in error_locations
    assert ["body", "requested_amount"] in error_locations
    assert response.status_code == 422