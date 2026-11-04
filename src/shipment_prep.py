
import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}


def clean_shipments(raw, medians=None):
    """
    Clean raw shipment data.

    Returns:
        clean: cleaned shipment DataFrame
        medians: dictionary containing historical median values
    """

    # Work on a copy so the original data is not changed
    clean = raw.copy()

    # 1. Remove exact duplicate rows
    clean = clean.drop_duplicates().copy()

    # 2. Normalize service_type
    clean["service_type"] = (
        clean["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # 3. Replace missing payment modes
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # 4. Convert declared_value from rupee text to numeric
    
    clean["declared_value_inr"] = pd.to_numeric(
    clean["declared_value"]
    .astype("string")
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
).astype("float64")


    # Remove the original text column
    clean = clean.drop(columns=["declared_value"])

    # 5. Convert dispatch_time to datetime
    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"],
        errors="coerce"
    )

    # 6. Treat unrealistic parcel weights as missing
    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # 7. Calculate medians from historical data,
    # or use previously saved medians for new shipments
    columns_to_impute = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:
        medians = {
            column: float(clean[column].median())
            for column in columns_to_impute
        }
    else:
        medians = {
            column: float(medians[column])
            for column in columns_to_impute
        }

    # 8. Fill missing numerical values
    for column in columns_to_impute:
        clean[column] = clean[column].fillna(medians[column])

    return clean, medians


def build_features(clean, hubs):
    """
    Merge hub information and create machine-learning features.
    Does not use delivery_hours to create features.
    """

    # Work on a copy
    features = clean.copy()

    # 1. Merge hub information
    hub_columns = [
        "hub_id",
        "hub_tier",
        "hub_daily_capacity"
    ]

    features = features.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # 2. Create promised delivery hours
    features["promised_hours"] = (
        features["service_type"]
        .map(PROMISED_HOURS)
    )

    # 3. Create COD indicator
    features["is_cod"] = (
        features["payment_mode"] == "COD"
    ).astype(int)

    # 4. Create weekend indicator
    features["is_weekend"] = (
        features["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # 5. Extract dispatch hour
    features["dispatch_hour"] = (
        features["dispatch_time"].dt.hour
    )

    # 6. Log-transform declared value
    features["log_declared_value"] = np.log1p(
        features["declared_value_inr"]
    )

    return features
