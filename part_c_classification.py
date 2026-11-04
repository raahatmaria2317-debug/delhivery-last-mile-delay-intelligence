
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. SETTINGS AND FEATURE COLUMNS
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

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA AND STRATIFIED SPLIT
# --------------------------------------------------

data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["is_delayed"].astype(int)

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

print("\nSPLIT CHECK")
print("Training rows:", len(X_train), "Delayed:", int(y_train.sum()))
print("Validation rows:", len(X_val), "Delayed:", int(y_val.sum()))
print("Test rows:", len(X_test), "Delayed:", int(y_test.sum()))

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41)


# --------------------------------------------------
# 3. METRIC HELPER
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# --------------------------------------------------
# 4. BLIND BASELINE AND THREE CLASSIFIERS
# --------------------------------------------------

model_rows = []

# Blind baseline: predict every parcel is on time.
blind_predictions = np.zeros(len(y_val), dtype=int)

blind_metrics = calculate_metrics(y_val, blind_predictions)
blind_metrics["model"] = "blind_all_on_time"
model_rows.append(blind_metrics)

# Logistic Regression with StandardScaler.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_val)

logistic_metrics = calculate_metrics(y_val, logistic_predictions)
logistic_metrics["model"] = "logistic_regression"
model_rows.append(logistic_metrics)

# Find the best decision tree depth using validation F1.
tree_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)

    tree_results.append({
        "depth": depth,
        "f1": f1_score(y_val, predictions, zero_division=0),
        "model": tree,
    })

best_tree_result = sorted(
    tree_results,
    key=lambda item: (-item["f1"], item["depth"]),
)[0]

best_depth = best_tree_result["depth"]
best_tree = best_tree_result["model"]

tree_metrics = calculate_metrics(
    y_val, best_tree.predict(X_val)
)
tree_metrics["model"] = f"decision_tree_depth_{best_depth}"
model_rows.append(tree_metrics)

# Default Random Forest.
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest.fit(X_train, y_train)
forest_predictions = default_forest.predict(X_val)

forest_metrics = calculate_metrics(y_val, forest_predictions)
forest_metrics["model"] = "random_forest_default"
model_rows.append(forest_metrics)

model_columns = [
    "model", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

pd.DataFrame(model_rows)[model_columns].to_csv(
    "outputs/part_c_models.csv", index=False
)


# --------------------------------------------------
# 5. FEATURE IMPORTANCE AND F-SCORES
# --------------------------------------------------

importance_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

importance_forest.fit(X_train, y_train)

selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": importance_forest.feature_importances_,
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv", index=False
)

noise_importance_share = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

print("\nNOISE FEATURE IMPORTANCE SHARE:", noise_importance_share)


# --------------------------------------------------
# 6. SELECTK EXPERIMENT
# --------------------------------------------------

selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ])

    selectk_model.fit(X_train, y_train)
    predictions = selectk_model.predict(X_val)

    selectk_rows.append({
        "k": k,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        ),
    })

pd.DataFrame(selectk_rows).to_csv(
    "outputs/part_c_selectk.csv", index=False
)


# --------------------------------------------------
# 7. SELECT EXACTLY 7 FEATURES FOR TUNING
# --------------------------------------------------

selector_seven = SelectKBest(score_func=f_classif, k=7)

X_train_7 = selector_seven.fit_transform(X_train, y_train)
X_val_7 = selector_seven.transform(X_val)


# --------------------------------------------------
# 8. GRID SEARCH AND RANDOMIZED SEARCH
# --------------------------------------------------

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

base_forest = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=base_forest,
    param_grid=parameter_grid,
    cv=5,
)

grid_search.fit(X_train_7, y_train)

grid_predictions = grid_search.predict(X_val_7)
grid_val_f1 = f1_score(
    y_val, grid_predictions, zero_division=0
)

random_search = RandomizedSearchCV(
    estimator=base_forest,
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)

random_search.fit(X_train_7, y_train)

random_predictions = random_search.predict(X_val_7)
random_val_f1 = f1_score(
    y_val, random_predictions, zero_division=0
)

# Untuned forest using the same 7 selected features.
default_seven_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_seven_forest.fit(X_train_7, y_train)

default_seven_predictions = default_seven_forest.predict(X_val_7)
default_seven_val_f1 = f1_score(
    y_val, default_seven_predictions, zero_division=0
)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": json.dumps(grid_search.best_params_, sort_keys=True),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1,
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": json.dumps(random_search.best_params_, sort_keys=True),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1,
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_seven_val_f1,
    },
]

tuning_columns = [
    "search", "candidates", "cv_folds", "total_fits",
    "best_params", "best_cv_accuracy", "val_f1",
]

tuning_table = pd.DataFrame(tuning_rows)[tuning_columns]

tuning_table.to_csv(
    "outputs/part_c_tuning.csv", index=False
)


# --------------------------------------------------
# 9. CHOOSE THE FINAL CONFIGURATION
# --------------------------------------------------

priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

winning_row = sorted(
    tuning_rows,
    key=lambda row: (-row["val_f1"], priority[row["search"]]),
)[0]

chosen_from = winning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            **chosen_params,
        ),
    ),
])

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 10. VALIDATION ALERT THRESHOLDS
# --------------------------------------------------

validation_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9,
]

threshold_rows = []

for threshold in threshold_values:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(y_val, predictions)
    metrics["threshold"] = threshold
    threshold_rows.append(metrics)

threshold_columns = [
    "threshold", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

threshold_table = pd.DataFrame(threshold_rows)[threshold_columns]

threshold_table.to_csv(
    "outputs/part_c_thresholds.csv", index=False
)

eligible_thresholds = threshold_table[
    threshold_table["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = float(
        eligible_thresholds["threshold"].max()
    )
else:
    chosen_threshold = 0.1

print("\nFINAL MODEL SELECTION")
print("Chosen from:", chosen_from)
print("Chosen parameters:", chosen_params)
print("Chosen alert threshold:", chosen_threshold)


# --------------------------------------------------
# 11. ONE-TIME FINAL TEST EVALUATION
# --------------------------------------------------

# Test rows are used only here, after all choices are complete.
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = calculate_metrics(y_test, test_predictions)

test_metrics["threshold"] = chosen_threshold

final_test_columns = [
    "threshold", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

pd.DataFrame(
    [test_metrics]
)[final_test_columns].to_csv(
    "outputs/part_c_final_test.csv", index=False
)


# --------------------------------------------------
# 12. SAVE FINAL CONFIGURATION
# --------------------------------------------------

max_depth_value = chosen_params["max_depth"]

if max_depth_value is None:
    max_depth_value = "unlimited"

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": max_depth_value,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv", index=False
)


# --------------------------------------------------
# 13. PRINT IMPORTANT RESULTS
# --------------------------------------------------

print("\nPART C COMPLETED")
print("\nMODEL COMPARISON")
print(pd.read_csv("outputs/part_c_models.csv").to_string(index=False))

print("\nFEATURE SCORES")
print(
    pd.read_csv("outputs/part_c_feature_scores.csv")
    .sort_values("f_rank")
    .to_string(index=False)
)

print("\nSELECT K RESULTS")
print(pd.read_csv("outputs/part_c_selectk.csv").to_string(index=False))

print("\nTUNING RESULTS")
print(pd.read_csv("outputs/part_c_tuning.csv").to_string(index=False))

print("\nTHRESHOLD RESULTS")
print(pd.read_csv("outputs/part_c_thresholds.csv").to_string(index=False))

print("\nFINAL TEST RESULTS")
print(pd.read_csv("outputs/part_c_final_test.csv").to_string(index=False))

print("\nFINAL CONFIGURATION")
print(pd.read_csv("outputs/part_c_final_config.csv").to_string(index=False))
