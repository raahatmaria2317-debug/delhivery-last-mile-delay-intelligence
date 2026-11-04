
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. SETTINGS AND FEATURES
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

DATA_PATH = Path("data/modelling_table.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD THE MODELLING TABLE
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)

X = data[FEATURES]
y = data["delivery_hours"]


# --------------------------------------------------
# 3. TRAIN, VALIDATION AND TEST SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)


# --------------------------------------------------
# 4. SCALING WITHOUT DATA LEAKAGE
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = float(
    np.abs(X_train_scaled.mean(axis=0)).max()
)

test_scaled_max_abs_mean = float(
    np.abs(X_test_scaled.mean(axis=0)).max()
)

checks = pd.DataFrame([
    ["train_rows", len(X_train)],
    ["val_rows", len(X_val)],
    ["test_rows", len(X_test)],
    ["train_scaled_max_abs_mean", train_scaled_max_abs_mean],
    ["test_scaled_max_abs_mean", test_scaled_max_abs_mean],
], columns=["metric", "value"])

checks.to_csv(OUTPUT_DIR / "part_b_checks.csv", index=False)


# --------------------------------------------------
# 5. METRIC CALCULATION FUNCTION
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


metrics_rows = []


def add_metrics(model_name, split_name, actual, predicted):
    scores = calculate_metrics(actual, predicted)

    metrics_rows.append({
        "model": model_name,
        "split": split_name,
        "MAE": scores["MAE"],
        "MSE": scores["MSE"],
        "RMSE": scores["RMSE"],
        "R2": scores["R2"]
    })


# --------------------------------------------------
# 6. MEAN BASELINE
# --------------------------------------------------

baseline_prediction = np.full(
    len(y_val),
    y_train.mean()
)

add_metrics(
    "mean_baseline",
    "val",
    y_val,
    baseline_prediction
)


# --------------------------------------------------
# 7. DECISION TREE DEPTH DIAGNOSTIC
# --------------------------------------------------

tree_depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_rows = []

for depth in tree_depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    tree_rows.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })


tree_depth_results = pd.DataFrame(tree_rows)

tree_depth_results.to_csv(
    OUTPUT_DIR / "part_b_tree_depth.csv",
    index=False
)


# Select the depth with the lowest validation RMSE.
# If there is a tie, prefer the smaller numerical depth.
best_tree_row = min(
    tree_rows,
    key=lambda row: (
        row["val_rmse"],
        float("inf") if row["max_depth"] == "unlimited"
        else row["max_depth"]
    )
)

best_depth = (
    None if best_tree_row["max_depth"] == "unlimited"
    else int(best_tree_row["max_depth"])
)


# --------------------------------------------------
# 8. FOUR CANDIDATE REGRESSION MODELS
# --------------------------------------------------

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


validation_rmse = {}

for model_name, model in models.items():

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    val_prediction = model.predict(X_val)

    add_metrics(
        model_name,
        "val",
        y_val,
        val_prediction
    )

    validation_rmse[model_name] = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )


# --------------------------------------------------
# 9. GRADIENT BOOSTING LEARNING-RATE EXPERIMENT
# --------------------------------------------------

learning_rates = [0.01, 0.1, 1.0]

learning_rate_rows = []

for lr in learning_rates:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    val_prediction = gb_model.predict(X_val)

    scores = calculate_metrics(y_val, val_prediction)

    learning_rate_rows.append({
        "learning_rate": lr,
        "val_rmse": scores["RMSE"],
        "val_r2": scores["R2"]
    })


pd.DataFrame(learning_rate_rows).to_csv(
    OUTPUT_DIR / "part_b_gb_learning_rate.csv",
    index=False
)


# --------------------------------------------------
# 10. RANK MODELS BY VALIDATION RMSE
# --------------------------------------------------

ranked_models = sorted(
    validation_rmse.items(),
    key=lambda item: item[1]
)

top_two_models = [
    model_name for model_name, rmse in ranked_models[:2]
]


# --------------------------------------------------
# 11. FIVE-FOLD CROSS-VALIDATION
# --------------------------------------------------

# Combine training and validation rows.
X_train_val = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0
)

# One shared KFold object for both models.
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_rows = []

for model_name in top_two_models:

    model = models[model_name]

    cv_scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_rows.append({
        "model": model_name,
        "mean_r2": cv_scores.mean(),
        "std_r2": cv_scores.std()
    })


pd.DataFrame(cv_rows).to_csv(
    OUTPUT_DIR / "part_b_cv.csv",
    index=False
)


# --------------------------------------------------
# 12. FINAL MODEL AND TEST EVALUATION
# --------------------------------------------------

# Select the candidate with the lowest validation RMSE.
final_model_name = ranked_models[0][0]
final_model = models[final_model_name]

# The final model is already fitted on training rows.
# Evaluate it on the test rows exactly once.
test_prediction = final_model.predict(X_test)

add_metrics(
    final_model_name,
    "test",
    y_test,
    test_prediction
)


# --------------------------------------------------
# 13. SAVE ALL MODEL METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metrics_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False
)


# --------------------------------------------------
# 14. PRINT RESULTS
# --------------------------------------------------

print("\n========== PART B REGRESSION RESULTS ==========")

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nScaling checks:")
print(checks.to_string(index=False))

print("\nDecision tree depth results:")
print(tree_depth_results.to_string(index=False))

print("\nBest decision tree depth:", best_depth)

print("\nValidation model ranking:")
for rank, (name, rmse) in enumerate(ranked_models, start=1):
    print(f"{rank}. {name}: RMSE = {rmse:.4f}")

print("\nGradient boosting learning-rate results:")
print(pd.DataFrame(learning_rate_rows).to_string(index=False))

print("\nCross-validation results:")
print(pd.DataFrame(cv_rows).to_string(index=False))

print("\nFinal model:", final_model_name)

print("\nFinal test metrics:")
print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))

print("\nAll Part B CSV files saved successfully.")
