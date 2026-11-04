
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import clean_shipments


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA AND FINAL CONFIGURATION
# --------------------------------------------------

data = pd.read_csv("data/modelling_table.csv")

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])

max_depth_value = config["max_depth"]

if pd.isna(max_depth_value) or str(max_depth_value).lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(max_depth_value)

threshold = float(config["threshold"])

X = data[FEATURES]
y = data["is_delayed"].astype(int)


# --------------------------------------------------
# 3. RECREATE THE SAME STRATIFIED DATA SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

assert len(X_train) == 840
assert len(X_val) == 180
assert len(X_test) == 180

assert int(y_train.sum()) == 195
assert int(y_val.sum()) == 42
assert int(y_test.sum()) == 41


# --------------------------------------------------
# 4. REBUILD THE FINAL PART C PIPELINE
# --------------------------------------------------

final_model = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
    )
])

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 5. SAVE THE MODEL
# --------------------------------------------------

joblib.dump(
    final_model,
    "models/delay_model.pkl"
)


# --------------------------------------------------
# 6. SAVE THE ALERT THRESHOLD
# --------------------------------------------------

pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    "models/delay_threshold.csv",
    index=False
)


# --------------------------------------------------
# 7. SAVE THE FEATURE ORDER
# --------------------------------------------------

pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    "models/feature_order.csv",
    index=False
)


# --------------------------------------------------
# 8. SAVE HISTORICAL IMPUTATION MEDIANS
# --------------------------------------------------

raw = pd.read_csv("data/raw_shipments.csv")

_, medians = clean_shipments(raw)

impute_values = pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
})

impute_values.to_csv(
    "models/impute_values.csv",
    index=False
)


# --------------------------------------------------
# 9. RELOAD THE SAVED MODEL
# --------------------------------------------------

loaded_model = joblib.load(
    "models/delay_model.pkl"
)


# --------------------------------------------------
# 10. COMPARE ORIGINAL AND RELOADED PREDICTIONS
# --------------------------------------------------

original_probabilities = final_model.predict_proba(X_test)[:, 1]

reloaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

max_difference = float(
    np.max(
        np.abs(
            original_probabilities - reloaded_probabilities
        )
    )
)

reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold",
    ],
    "value": [
        len(X_test),
        max_difference,
        threshold,
    ]
})

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\nMODEL PACKAGING COMPLETED")

print("\nFINAL CONFIGURATION")
print(config.to_string())

print("\nSAVED MODEL FILES")
print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")

print("\nRELOAD CHECK")
print(reload_check.to_string(index=False))

print("\nMaximum probability difference:", max_difference)

if max_difference <= 1e-9:
    print("PASS: Reloaded model predictions match.")
else:
    print("WARNING: Reloaded model predictions differ.")

print("\nPart D model packaging completed.")
