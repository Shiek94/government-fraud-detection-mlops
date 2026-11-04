import pandas as pd
from pathlib import Path
from data_generation import generate_applications


# Generate 12 different month worth of synthetic data with differing seeds and request_multipliers
def generate_yearly_data():
    # Initialize an empty df list
    df_list = []
    for month_number in range(1, 13):
        month = f"2026-{month_number:02d}"
        random_seed = month_number
        if month_number <= 6:
            request_multiplier = 1.0
        else:
            request_multiplier = 1.5
        monthly_df = generate_applications(seed=random_seed, month=month, request_multiplier=request_multiplier)
        df_list.append(monthly_df)

    # Combine 12 monthly df into yearly df
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    # Generate yearly df
    combined_df = generate_yearly_data()

    print(combined_df)
    print(combined_df.groupby("submitted_month")["requested_amount"].mean())

    assert combined_df.shape == (12000, 8)
    assert combined_df["application_id"].is_unique
    assert combined_df["submitted_month"].nunique() == 12

    # Save yearly df to csv -> data directory
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    data_dir.mkdir(exist_ok=True)

    combined_df.to_csv(
        data_dir / "synthetic_applications_2026.csv",
        index=False
    )


