
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():

    # Create outputs folder if it does not exist
    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    modelling = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # --------------------------------------------------
    # 2. CREATE HUB PROFILES
    # --------------------------------------------------

    hub_profile = (
        modelling.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    hub_profile.to_csv(
        "outputs/hub_profile.csv",
        index=False
    )

    print("\nHUB PROFILE")
    print(hub_profile.to_string(index=False))

    # --------------------------------------------------
    # 3. SCALE HUB PROFILE FEATURES
    # --------------------------------------------------

    profile_columns = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        hub_profile[profile_columns]
    )

    # --------------------------------------------------
    # 4. COMPARE CLUSTER SEPARATION FOR K = 2, 3, 4, 5
    # --------------------------------------------------

    separation_results = []

    for k in [2, 3, 4, 5]:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(scaled_data)

        within_distances = []
        across_distances = []

        # Calculate distances for all 276 hub pairs
        for i, j in combinations(range(len(scaled_data)), 2):

            distance = np.linalg.norm(
                scaled_data[i] - scaled_data[j]
            )

            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)

        mean_within = np.mean(within_distances)
        mean_across = np.mean(across_distances)

        ratio = mean_across / mean_within

        cluster_sizes = sorted(
            pd.Series(labels).value_counts().tolist(),
            reverse=True
        )

        cluster_sizes_text = "-".join(
            str(size) for size in cluster_sizes
        )

        separation_results.append({
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": cluster_sizes_text
        })

    separation = pd.DataFrame(separation_results)

    separation.to_csv(
        "outputs/part_d_separation.csv",
        index=False
    )

    print("\nCLUSTER SEPARATION")
    print(separation.to_string(index=False))

    # --------------------------------------------------
    # 5. FINAL K-MEANS MODEL WITH K = 3
    # --------------------------------------------------

    final_kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_labels = final_kmeans.fit_predict(
        scaled_data
    )

    hub_profile["cluster"] = cluster_labels

    # --------------------------------------------------
    # 6. NAME CLUSTERS USING THEIR DELAY RATES
    # --------------------------------------------------

    cluster_delay_rates = (
        hub_profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )

    cluster_names = {
        cluster_delay_rates.index[0]: "Standard",
        cluster_delay_rates.index[1]: "Watch",
        cluster_delay_rates.index[2]: "Escalate",
    }

    hub_profile["segment"] = (
        hub_profile["cluster"].map(cluster_names)
    )

    # --------------------------------------------------
    # 7. CREATE HUB SEGMENTS FILE
    # --------------------------------------------------

    hub_segments = hub_profile[
        ["hub_id", "segment"]
    ].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id").reset_index(drop=True)

    hub_segments.to_csv(
        "outputs/hub_segments.csv",
        index=False
    )

    print("\nHUB SEGMENTS")
    print(hub_segments.to_string(index=False))

    # --------------------------------------------------
    # 8. CREATE SEGMENT PROFILE
    # --------------------------------------------------

    segment_profile = (
        hub_profile.groupby("segment", as_index=False)
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
    )

    segment_order = ["Standard", "Watch", "Escalate"]

    segment_profile["segment"] = pd.Categorical(
        segment_profile["segment"],
        categories=segment_order,
        ordered=True
    )

    segment_profile = (
        segment_profile.sort_values("segment")
        .reset_index(drop=True)
    )

    segment_profile["segment"] = (
        segment_profile["segment"].astype(str)
    )

    segment_profile.to_csv(
        "outputs/segment_profile.csv",
        index=False
    )

    print("\nSEGMENT PROFILE")
    print(segment_profile.to_string(index=False))

    # --------------------------------------------------
    # 9. FIND TIER VS SEGMENT MISMATCHES
    # --------------------------------------------------

    expected_segments = {
        1: "Standard",
        2: "Watch",
        3: "Escalate",
    }

    hub_segments["expected_segment"] = (
        hub_segments["hub_tier"].map(expected_segments)
    )

    mismatches = hub_segments[
        hub_segments["segment"] != hub_segments["expected_segment"]
    ][
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].copy()

    mismatches.to_csv(
        "outputs/tier_vs_segment_mismatch.csv",
        index=False
    )

    print("\nTIER VS SEGMENT MISMATCHES")
    print(mismatches.to_string(index=False))

    print("\nPart D segmentation completed successfully.")
    print("All five output files have been saved in the outputs folder.")


if __name__ == "__main__":
    main()
