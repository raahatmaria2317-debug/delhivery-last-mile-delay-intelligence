
import os
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features, PROMISED_HOURS


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

os.makedirs("outputs", exist_ok=True)

raw_rows = len(raw)

# Count exact duplicate rows
duplicate_rows = int(raw.duplicated().sum())

# Record missing values before cleaning
# Count missing values after removing exact duplicates
raw_without_duplicates = raw.drop_duplicates()

missing_distance = int(raw_without_duplicates["distance_km"].isna().sum())
missing_weight = int(raw_without_duplicates["parcel_weight_kg"].isna().sum())
missing_value = int(raw_without_duplicates["declared_value"].isna().sum())
missing_payment = int(raw_without_duplicates["payment_mode"].isna().sum())

# Count unrealistic weight values
sentinel_weights = int((raw["parcel_weight_kg"] > 100).sum())

# Count service type spellings before normalization
service_spellings_before = int(raw["service_type"].nunique())

# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------

clean, medians = clean_shipments(raw)

service_spellings_after = int(clean["service_type"].nunique())

# --------------------------------------------------
# 3. BUILD FEATURES
# --------------------------------------------------

modelling = build_features(clean, hubs)

# Create target variable
modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)

# Check missing values in features and target
feature_columns = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops", "rain_flag",
    "sale_event_flag", "is_cod", "is_weekend",
    "dispatch_hour", "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no", "delivery_hours",
    "is_delayed"
]

missing_after = int(
    modelling[feature_columns].isna().sum().sum()
)

# Save modelling table
modelling.to_csv(
    "data/modelling_table.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 4. DATA QUALITY CHECKS
# --------------------------------------------------

q1 = float(modelling["delivery_hours"].quantile(0.25))
q3 = float(modelling["delivery_hours"].quantile(0.75))

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = int(
    (
        (modelling["delivery_hours"] < lower_bound)
        | (modelling["delivery_hours"] > upper_bound)
    ).sum()
)

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows),
    ("rows_after_dedup", len(clean)),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_payment),
    ("sentinel_weight_rows", sentinel_weights),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("rows_after_hub_merge", len(modelling)),
    ("nulls_in_features_and_target", missing_after),
    ("delayed_count", int(modelling["is_delayed"].sum())),
    ("delay_rate", float(modelling["is_delayed"].mean())),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    ("hours_outlier_count", outlier_count),
    ("declared_value_mean_inr", float(modelling["declared_value_inr"].mean())),
    ("declared_value_median_inr", float(modelling["declared_value_inr"].median())),
    ("log_declared_value_mean", float(modelling["log_declared_value"].mean())),
    ("log_declared_value_median", float(modelling["log_declared_value"].median()))
]

checks_df = pd.DataFrame(checks, columns=["check", "value"])

checks_df.to_csv(
    "outputs/part_a_checks.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 5. DELAY BY SERVICE TYPE
# --------------------------------------------------

service_delay = (
    modelling.groupby("service_type")
    .agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

service_delay["group_type"] = "service_type"
service_delay["group"] = service_delay["service_type"]

service_delay = service_delay[
    ["group_type", "group", "shipments", "delayed", "delay_rate"]
]

# --------------------------------------------------
# 6. DELAY BY HUB TIER
# --------------------------------------------------

tier_delay = (
    modelling.groupby("hub_tier")
    .agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

tier_delay["group_type"] = "hub_tier"
tier_delay["group"] = tier_delay["hub_tier"].astype(str)

tier_delay = tier_delay[
    ["group_type", "group", "shipments", "delayed", "delay_rate"]
]

# --------------------------------------------------
# 7. DELAY BY DISTANCE GROUP
# --------------------------------------------------

distance_bins = [0, 200, 400, 600, 2000]

distance_labels = [
    "0-200",
    "201-400",
    "401-600",
    "601+"
]

modelling["distance_group"] = pd.cut(
    modelling["distance_km"],
    bins=distance_bins,
    labels=distance_labels,
    include_lowest=True,
    right=True
)

distance_delay = (
    modelling.groupby("distance_group", observed=False)
    .agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

distance_delay["group_type"] = "distance_group"
distance_delay["group"] = distance_delay["distance_group"].astype(str)

distance_delay = distance_delay[
    ["group_type", "group", "shipments", "delayed", "delay_rate"]
]

# Combine all group analyses
delay_by_group = pd.concat(
    [service_delay, tier_delay, distance_delay],
    ignore_index=True
)

delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 8. CORRELATION MATRIX
# --------------------------------------------------

numeric_data = modelling.select_dtypes(include=np.number)

correlations = numeric_data.corr()

correlations.to_csv(
    "outputs/part_a_correlations.csv",
    index=False,
    float_format="%.4f"
)

# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\nPART A COMPLETED SUCCESSFULLY")

print("\nData quality checks:")
print(checks_df.to_string(index=False))

print("\nDelay by group:")
print(delay_by_group.to_string(index=False))

print("\nTop correlations with delivery_hours:")
print(
    correlations["delivery_hours"]
    .drop("delivery_hours")
    .abs()
    .sort_values(ascending=False)
    .head(5)
)

print("\nGenerated files:")
print("data/modelling_table.csv")
print("outputs/part_a_checks.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")
