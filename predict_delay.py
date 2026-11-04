import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


def main():
    # 1. Load the new shipment data and hub information
    raw_new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # 2. Load the saved model and its supporting files
    model = joblib.load("models/delay_model.pkl")

    threshold = float(
        pd.read_csv("models/delay_threshold.csv").loc[0, "threshold"]
    )

    feature_order = pd.read_csv(
        "models/feature_order.csv"
    )["feature"].tolist()

    impute_table = pd.read_csv("models/impute_values.csv")

    medians = dict(
        zip(impute_table["column"], impute_table["median"])
    )

    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    # 3. Clean the new shipments using the saved training medians
    clean, _ = clean_shipments(raw_new, medians=medians)

    # 4. Build features using the shared preprocessing function
    features = build_features(clean, hubs)

    # 5. Arrange features in exactly the order used during training
    X_new = features[feature_order]

    # 6. Predict the probability of delay
    raw_probabilities = model.predict_proba(X_new)[:, 1]

    # Flag shipments using the saved threshold
    flagged = (raw_probabilities >= threshold).astype(int)

    # Round probabilities for the output file
    delay_probability = np.round(raw_probabilities, 4)

    # 7. Create the shipment prediction table
    results = pd.DataFrame({
        "shipment_id": features["shipment_id"],
        "hub_id": features["hub_id"],
        "delay_probability": delay_probability,
        "flagged": flagged,
    })

    # Preserve the original shipment order when adding hub segments
    results["_row_order"] = np.arange(len(results))

    results = results.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        validate="many_to_one",
        sort=False
    )

    results = (
        results.sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )

    # 8. Assign the recommended action
    conditions = [
        (results["flagged"] == 1) &
        (results["segment"] == "Escalate"),

        results["flagged"] == 1
    ]

    choices = [
        "Escalate to hub manager",
        "Proactive customer call"
    ]

    results["action"] = np.select(
        conditions,
        choices,
        default="No action"
    )

    # 9. Arrange columns in the required order
    results = results[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action"
        ]
    ]

    # 10. Check the output before saving
    assert len(results) == len(raw_new), "Shipment row count changed."
    assert len(results) == 48, "Expected exactly 48 new shipments."
    assert results["shipment_id"].is_unique, "Duplicate shipment IDs found."
    assert not results.isna().any().any(), "Missing values found."

    # Check that every shipment has the correct action
    escalate_mask = (
        (results["flagged"] == 1) &
        (results["segment"] == "Escalate")
    )

    proactive_mask = (
        (results["flagged"] == 1) &
        (results["segment"] != "Escalate")
    )

    no_action_mask = results["flagged"] == 0

    assert (
        results.loc[escalate_mask, "action"] ==
        "Escalate to hub manager"
    ).all()

    assert (
        results.loc[proactive_mask, "action"] ==
        "Proactive customer call"
    ).all()

    assert (
        results.loc[no_action_mask, "action"] ==
        "No action"
    ).all()

    # 11. Save the final output
    results.to_csv(
        "outputs/new_shipment_actions.csv",
        index=False,
        float_format="%.4f",
        lineterminator="\n"
    )

    # 12. Display a summary
    print("\nNEW SHIPMENT PREDICTIONS COMPLETED")
    print("----------------------------------")
    print("Total shipments:", len(results))
    print("Flagged shipments:", int(results["flagged"].sum()))
    print("Hub manager escalations:", int(escalate_mask.sum()))
    print("Proactive customer calls:", int(proactive_mask.sum()))
    print("No action:", int(no_action_mask.sum()))

    print("\nACTION COUNTS")
    print(results["action"].value_counts())

    print("\nFIRST 10 PREDICTIONS")
    print(results.head(10).to_string(index=False))

    print("\nSaved: outputs/new_shipment_actions.csv")
    print("All checks passed.")


if __name__ == "__main__":
    main()