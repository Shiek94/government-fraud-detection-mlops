# Generation of a synthetic dataset for fraud detection

import numpy as np
import pandas as pd

def generate_applications(n_samples=1000, seed=42, month="2026-01", request_multiplier=1.0):
    # Initialization of randomizer object
    rng = np.random.default_rng(seed)

    # Dataframe column data creation
    cleaned_month = month.replace("-", "")
    application_id = [f"APP-{cleaned_month}-{i:05d}" for i in range(1, n_samples + 1)]

    monthly_income = rng.integers(low=0, high=6001, size=n_samples)
    requested_amount = rng.integers(low=200, high=10001, size=n_samples)
    requested_amount = (requested_amount * request_multiplier).astype(int)
    household_size = rng.integers(low=1, high=7, size=n_samples)
    previous_applications = rng.integers(low=0, high=6, size=n_samples)

    employment_stati = ["employed", "unemployed", "self-employed"]
    employment_status = rng.choice(employment_stati, size=n_samples)

    # Create Dataframe with column names and fill with data
    applications_df = pd.DataFrame({
        "application_id": application_id,
        "monthly_income": monthly_income,
        "requested_amount": requested_amount,
        "household_size": household_size,
        "previous_applications": previous_applications,
        "employment_status": employment_status,
        "submitted_month": month
    }
    )

    # Fraud probability randomization process
    fraud_probabilities = np.full(n_samples, 0.05)
    random_values = rng.random(n_samples)

    repeat_applicant_mask = applications_df["previous_applications"] >= 3
    high_requester_mask = applications_df["requested_amount"] > (applications_df["monthly_income"] * 2)

    fraud_probabilities[repeat_applicant_mask] += 0.1
    fraud_probabilities[high_requester_mask] += 0.15

    fraud_confirmed = (random_values < fraud_probabilities).astype(int)

    applications_df["fraud_confirmed"] = fraud_confirmed

    return applications_df

if __name__ == "__main__":
    df1 = generate_applications(1000, 42, month="2026-03")
    df2 = generate_applications(1000, 42, month="2026-03")
    df3 = generate_applications(1200, 43)
    df4 = generate_applications(1200, 42)
    january = generate_applications(n_samples=1000, seed=42, month="2026-01")
    february = generate_applications(n_samples=1200, seed=43, month="2026-02")
    baseline = generate_applications(
        n_samples=1000,
        seed=42,
        month="2026-01",
        request_multiplier=1.0
    )

    shifted = generate_applications(
        n_samples=1000,
        seed=42,
        month="2026-07",
        request_multiplier=1.5
    )

    assert df1.shape == (1000, 8)
    assert df3.shape == (1200, 8)
    assert df1.equals(df2)
    assert not df3.equals(df4)
    assert shifted["requested_amount"].mean() > baseline["requested_amount"].mean()

    print(january.shape)
    print(february.shape)
    print(january["application_id"].head())
    print(february["application_id"].head())
    print(baseline["requested_amount"].mean())
    print(shifted["requested_amount"].mean())














