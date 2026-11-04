# Delhivery Last-Mile Delay Intelligence Pipeline

An end-to-end machine learning project for last-mile delivery intelligence.

## Objectives
- Predict parcel delivery duration.
- Identify potentially delayed shipments before delivery.
- Segment 24 delivery hubs by operational performance.
- Generate proactive actions for new shipments.

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Git and GitHub

## Dataset
This educational project uses a synthetic shipment dataset. It does not contain actual Delhivery operational or customer records.

## Project Status
Under development — AI/ML Certification Program, E&ICT Academy, IIT Roorkee.


## Model results (regression)

### 1. Baseline performance

The mean baseline achieved a validation R² of **-0.0212**. This means that predicting the average delivery time from the training data performs slightly worse than predicting the average delivery time of the validation data for every parcel. This confirms that a useful regression model is needed.

### 2. Model comparison

Four regression models were evaluated on the validation dataset.

| Model | Validation RMSE (hours) | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Gradient Boosting | 4.9165 | 0.9306 |
| Random Forest | 5.4680 | 0.9141 |
| Decision Tree | 6.2796 | 0.8867 |

**Linear Regression was the winning model**, achieving the lowest validation RMSE of **4.8912 hours**. Its performance was slightly better than Gradient Boosting.

### 3. Decision tree depth analysis

The decision tree was tested with depths of 1, 2, 3, 4, 6, 8, 12 and unlimited depth.

- **High bias:** Depth 1 had a training RMSE of 11.63 and validation RMSE of 12.13. Both errors were high, indicating that the tree was too simple to learn the underlying patterns.
- **Best depth:** Depth 6 achieved the lowest validation RMSE of **6.2796 hours**.
- **High variance:** The unlimited-depth tree achieved zero training RMSE but a validation RMSE of 7.7890. The large gap indicates overfitting because the tree memorised the training data but did not generalise as well to unseen data.

### 4. Gradient boosting learning-rate experiment

Three learning rates were compared while keeping the number of estimators at 100.

| Learning Rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3609 | 0.8444 |

The learning rate of **0.1 performed best**. A rate of 0.01 learned too slowly with only 100 estimators, while 1.0 was too aggressive and overshot useful improvements, resulting in weaker validation performance.

### 5. Cross-validation comparison

The two best validation models were evaluated using the same shuffled five-fold cross-validation configuration on the combined training and validation data.

| Model | Mean R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9352 | 0.0044 |

Gradient Boosting achieved a slightly higher cross-validation mean R². However, Linear Regression remained the final choice because it had the lowest validation RMSE, as required by the project selection rule. Both models showed consistent performance across the five folds.

### 6. Final test performance

The selected Linear Regression model was evaluated once on the untouched test dataset.

| Metric | Test Result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4625 |
| RMSE | 4.5236 hours |
| R² | 0.9493 |

The final test R² of **0.9493** indicates that the model explains approximately 94.93% of the variation in delivery hours on the test dataset.

### 7. Conclusion

Linear Regression was selected as the final regression model because it achieved the lowest validation RMSE and strong final test performance. The experiment also demonstrated the importance of separating training, validation and test data, fitting the scaler only on training data, and using cross-validation to assess model stability.


## Model results (classification)

### 1. Classification objective

The objective of Part C was to predict whether a shipment would be delayed using the `is_delayed` target column. The task was treated as a binary classification problem:

- `0`: On-time shipment
- `1`: Delayed shipment

The same 17 features used in the regression task were used for classification.

### 2. Data splitting

The dataset was divided using stratified splitting to preserve the proportion of delayed and on-time shipments.

| Dataset | Rows | Delayed shipments |
|---|---:|---:|
| Training | 840 | 195 |
| Validation | 180 | 42 |
| Test | 180 | 41 |

The random state was set to 42.

### 3. Model comparison

Four models were evaluated on the validation dataset.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Blind all-on-time baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest (default) | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The blind baseline achieved 76.67% accuracy by predicting every shipment as on-time. However, it failed to identify any delayed shipments. This demonstrates why accuracy alone is not sufficient for evaluating a classification model when the target classes are imbalanced.

Logistic Regression achieved the highest F1-score among the initial model comparisons.

### 4. Feature selection

Feature importance was examined using Random Forest importance and F-scores from `SelectKBest(f_classif)`.

The seven highest-ranked features were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

All five noise features ranked outside the top seven. Their combined Random Forest importance was approximately 22.44%.

Different feature counts were tested:

| Number of features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Using seven features produced the highest validation F1-score.

### 5. Hyperparameter tuning

Grid Search and Randomized Search were performed using five-fold cross-validation.

| Method | Candidates | Total fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Random Forest (7 features) | 1 | 1 | — | 0.8333 |

Although Grid Search achieved slightly higher cross-validation accuracy than Randomized Search, neither tuned model outperformed the default Random Forest on the validation F1-score.

Therefore, the default Random Forest using seven selected features was chosen. This also demonstrates that hyperparameter tuning does not always improve performance, particularly when validation data is limited.

### 6. Classification threshold selection

The default classification threshold of 0.5 was evaluated alongside thresholds from 0.1 to 0.9.

The selection rule was to choose the highest threshold that maintained validation recall of at least 0.80.

The selected threshold was **0.5**.

At this threshold, validation performance was:

- Accuracy: 0.9111
- Precision: 0.7955
- Recall: 0.8333
- F1-score: 0.8140

Increasing the threshold generally reduces false positives but can also increase false negatives. The selected threshold balances identifying delayed shipments with avoiding unnecessary delay alerts.

### 7. Final test performance

The selected model and threshold were evaluated once on the untouched test dataset.

| Metric | Test result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |

The test confusion matrix was:

| | Predicted on-time | Predicted delayed |
|---|---:|---:|
| Actual on-time | 125 | 14 |
| Actual delayed | 10 | 31 |

Test recall was lower than validation recall. One reason to interpret this difference carefully is that the test set contained only 41 delayed shipments, so a relatively small number of additional missed delays can noticeably change the recall score.

### 8. Final selected configuration

- Feature selection: SelectKBest with k = 7
- Model: Random Forest Classifier
- Number of estimators: 100
- Maximum depth: Unlimited (None)
- Minimum samples per leaf: 1
- Classification threshold: 0.5
- Selected model: Default Random Forest using seven features

The final test results indicate that the model can identify delayed shipments reasonably well, while also showing the importance of evaluating precision and recall rather than relying on accuracy alone.


# Delhivery Last-Mile Delay Intelligence Pipeline

An end-to-end machine learning project for last-mile delivery intelligence.

## Objectives

- Predict parcel delivery duration.
- Identify potentially delayed shipments before delivery.
- Segment 24 delivery hubs by operational performance.
- Generate proactive actions for new shipments.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Git and GitHub

## Dataset

This educational project uses a synthetic shipment dataset. It does not contain actual Delhivery operational or customer records.

## Project Status

Completed project — AI/ML Certification Program, E&ICT Academy, IIT Roorkee.

---

# 1. How to Run

Run the following commands from the repository's main folder.

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate the dataset

```bash
python generate_data.py
```

### Step 3: Run data cleaning and exploratory analysis

```bash
python part_a_clean_eda.py
```

### Step 4: Run delivery-time regression

```bash
python part_b_regression.py
```

### Step 5: Run delay classification

```bash
python part_c_classification.py
```

### Step 6: Run hub segmentation

```bash
python part_d_segments.py
```

### Step 7: Package the final classification model

```bash
python package_model.py
```

This trains and saves the final classification model and its supporting files in the `models/` folder.

### Step 8: Generate actions for new shipments

```bash
python predict_delay.py
```

The final predictions are saved to:

`outputs/new_shipment_actions.csv`

The prediction script uses the saved model, preprocessing functions, imputation values, feature order, alert threshold and hub segments.

---

# 2. Data Findings

The initial dataset contained 1,224 shipment rows. After cleaning and removing duplicate records, 1,200 shipment rows remained.

Important preprocessing steps included:

- Removing duplicate records.
- Standardizing service-type values.
- Handling missing payment-mode values.
- Converting declared values into numeric values.
- Converting dispatch timestamps into datetime values.
- Treating parcel weights above 100 kg as invalid values.
- Imputing missing distance, parcel weight and declared value using training-data medians.
- Combining shipment information with hub information.

### Imputation Medians

| Feature | Median |
|---|---:|
| Distance | 314.95 km |
| Parcel weight | 1.83 kg |
| Declared value | ₹1,121.50 |

The cleaned dataset was used to build the modelling table for regression and classification.

---

# 3. Model Results (Regression)

## 3.1 Baseline Performance

The mean baseline achieved a validation R² of **-0.0212**. This means that predicting the average delivery time from the training data performs slightly worse than predicting the average delivery time of the validation data for every parcel. This confirms that a useful regression model is needed.

## 3.2 Model Comparison

Four regression models were evaluated on the validation dataset.

| Model | Validation RMSE (hours) | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Gradient Boosting | 4.9165 | 0.9306 |
| Random Forest | 5.4680 | 0.9141 |
| Decision Tree | 6.2796 | 0.8867 |

**Linear Regression was the winning model**, achieving the lowest validation RMSE of **4.8912 hours**. Its performance was slightly better than Gradient Boosting.

## 3.3 Decision Tree Depth Analysis

The decision tree was tested with depths of 1, 2, 3, 4, 6, 8, 12 and unlimited depth.

- **High bias:** Depth 1 had a training RMSE of 11.63 and validation RMSE of 12.13. Both errors were high, indicating that the tree was too simple to learn the underlying patterns.
- **Best depth:** Depth 6 achieved the lowest validation RMSE of **6.2796 hours**.
- **High variance:** The unlimited-depth tree achieved zero training RMSE but a validation RMSE of 7.7890. The large gap indicates overfitting because the tree memorised the training data but did not generalise as well to unseen data.

## 3.4 Gradient Boosting Learning-Rate Experiment

Three learning rates were compared while keeping the number of estimators at 100.

| Learning Rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3609 | 0.8444 |

The learning rate of **0.1 performed best**. A rate of 0.01 learned too slowly with only 100 estimators, while 1.0 was too aggressive and resulted in weaker validation performance.

## 3.5 Cross-Validation Comparison

The two best validation models were evaluated using the same shuffled five-fold cross-validation configuration on the combined training and validation data.

| Model | Mean R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9352 | 0.0044 |

Gradient Boosting achieved a slightly higher cross-validation mean R². However, Linear Regression remained the final choice because it had the lowest validation RMSE, as required by the project selection rule. Both models showed consistent performance across the five folds.

## 3.6 Final Test Performance

The selected Linear Regression model was evaluated once on the untouched test dataset.

| Metric | Test Result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4625 |
| RMSE | 4.5236 hours |
| R² | 0.9493 |

The final test R² of **0.9493** indicates that the model explains approximately 94.93% of the variation in delivery hours on the test dataset.

## 3.7 Conclusion

Linear Regression was selected as the final regression model because it achieved the lowest validation RMSE and strong final test performance.

The experiment also demonstrated the importance of separating training, validation and test data, fitting the scaler only on training data, and using cross-validation to assess model stability.

---

# 4. Model Results (Classification)

## 4.1 Classification Objective

The objective of Part C was to predict whether a shipment would be delayed using the `is_delayed` target column.

The task was treated as a binary classification problem:

- `0`: On-time shipment
- `1`: Delayed shipment

The same 17 features used in the regression task were used for classification.

## 4.2 Data Splitting

The dataset was divided using stratified splitting to preserve the proportion of delayed and on-time shipments.

| Dataset | Rows | Delayed Shipments |
|---|---:|---:|
| Training | 840 | 195 |
| Validation | 180 | 42 |
| Test | 180 | 41 |

The random state was set to 42.

## 4.3 Model Comparison

Four models were evaluated on the validation dataset.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Blind all-on-time baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest (default) | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The blind baseline achieved 76.67% accuracy by predicting every shipment as on-time. However, it failed to identify any delayed shipments.

This demonstrates why accuracy alone is not sufficient for evaluating a classification model when the target classes are imbalanced.

Logistic Regression achieved the highest F1-score among the initial model comparisons.

## 4.4 Feature Selection

Feature importance was examined using Random Forest importance and F-scores from `SelectKBest(f_classif)`.

The seven highest-ranked features were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

All five noise features ranked outside the top seven. Their combined Random Forest importance was approximately 22.44%.

Different feature counts were tested:

| Number of Features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Using seven features produced the highest validation F1-score.

## 4.5 Hyperparameter Tuning

Grid Search and Randomized Search were performed using five-fold cross-validation.

| Method | Candidates | Total Fits | Best CV Accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Random Forest (7 features) | 1 | 1 | — | 0.8333 |

Although Grid Search achieved slightly higher cross-validation accuracy than Randomized Search, neither tuned model outperformed the default Random Forest on the validation F1-score.

Therefore, the default Random Forest using seven selected features was chosen.

This demonstrates that hyperparameter tuning does not always improve performance, particularly when validation data is limited.

## 4.6 Classification Threshold Selection

The default classification threshold of 0.5 was evaluated alongside thresholds from 0.1 to 0.9.

The selection rule was to choose the highest threshold that maintained validation recall of at least 0.80.

The selected threshold was **0.5**.

At this threshold, validation performance was:

- Accuracy: 0.9111
- Precision: 0.7955
- Recall: 0.8333
- F1-score: 0.8140

Increasing the threshold generally reduces false positives but can also increase false negatives.

The selected threshold balances identifying delayed shipments with avoiding unnecessary delay alerts.

## 4.7 Final Test Performance

The selected model and threshold were evaluated once on the untouched test dataset.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |

The test confusion matrix was:

| | Predicted On-time | Predicted Delayed |
|---|---:|---:|
| Actual On-time | 125 | 14 |
| Actual Delayed | 10 | 31 |

Test recall was lower than validation recall. One reason to interpret this difference carefully is that the test set contained only 41 delayed shipments, so a relatively small number of additional missed delays can noticeably change the recall score.

## 4.8 Final Selected Configuration

- Feature selection: SelectKBest with k = 7
- Model: Random Forest Classifier
- Number of estimators: 100
- Maximum depth: Unlimited (None)
- Minimum samples per leaf: 1
- Classification threshold: 0.5
- Selected model: Default Random Forest using seven features

The final test results indicate that the model can identify delayed shipments reasonably well, while also showing the importance of evaluating precision and recall rather than relying on accuracy alone.

---

# 5. Hub Segmentation Results (Part D)

## 5.1 Segmentation Objective

The objective of Part D was to group the 24 delivery hubs according to their operational performance.

The segmentation uses hub-level delivery and operational characteristics to identify hubs that require different levels of attention.

The analysis considered:

- Mean delivery time in hours
- Delay rate
- Mean delivery distance
- Mean number of stops

Each feature was scaled before applying K-means clustering so that features with larger numerical ranges would not dominate the clustering process.

## 5.2 Choosing the Number of Clusters

K-means clustering was evaluated using different values of K. Cluster separation was compared using the ratio of between-cluster variation to within-cluster variation.

| Number of Clusters (K) | Separation Ratio | Cluster Sizes |
|---:|---:|---|
| 2 | 2.8940 | 15, 9 |
| 3 | 5.3928 | 9, 8, 7 |
| 4 | 6.3047 | 8, 7, 5, 4 |
| 5 | 6.8866 | 8, 7, 4, 3, 2 |

Although the separation ratio increased when K increased to 4 and 5, **K = 3 was selected**.

The reason was operational usefulness. With three clusters, every segment contains at least seven hubs. With four or five clusters, some groups contain fewer than six hubs. These smaller groups would be less suitable for creating separate operational playbooks.

## 5.3 Final Hub Segments

The 24 hubs were assigned to three operational segments:

| Segment | Number of Hubs | Mean Delivery Time (Hours) | Mean Delay Rate | Mean Distance (km) | Mean Stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% | 188.93 | 55.37 |
| Watch | 8 | 56.81 | 15.50% | 314.18 | 70.72 |
| Escalate | 9 | 64.65 | 46.00% | 470.48 | 84.34 |

### Standard

These hubs have the lowest average delivery time and delay rate. They should continue with normal operational monitoring and existing delivery procedures.

### Watch

These hubs have higher delivery times and delay rates than Standard hubs. They should receive closer monitoring, with attention to route efficiency, delivery stops and emerging delays.

### Escalate

These hubs have the highest average delivery time, delay rate, distance and number of stops. They should receive priority operational attention and hub-level investigation.

## 5.4 Hub Tier Versus Segment Mismatches

The operational segment was compared with the existing hub tier to identify hubs whose observed performance may not match their assigned tier.

Four mismatches were identified:

| Hub | Assigned Segment | City | Existing Tier |
|---|---|---|---|
| HUB07 | Watch | Kolkata | Tier 1 |
| HUB13 | Escalate | Nagpur | Tier 2 |
| HUB15 | Escalate | Kochi | Tier 2 |
| HUB22 | Watch | Dehradun | Tier 3 |

These mismatches indicate that the existing hub tier alone does not fully describe operational performance.

For example, HUB07 is a Tier 1 hub but falls into the Watch segment. HUB13 and HUB15 are Tier 2 hubs but fall into the Escalate segment.

These hubs should be reviewed using their actual delivery performance rather than relying only on their existing tier classification.

## 5.5 Operational Interpretation

The three segments provide a practical way to prioritize hub-level attention:

- **Standard:** Maintain normal operations and routine monitoring.
- **Watch:** Investigate emerging delays and monitor operational performance more frequently.
- **Escalate:** Prioritize management review and investigate possible capacity, distance and routing problems.

The segmentation is intended to support operational decisions, not to permanently label hubs. Segment assignments should be reviewed as new delivery data becomes available.

## 5.6 Conclusion

K-means clustering divided the 24 hubs into three operationally useful groups: Standard, Watch and Escalate.

K = 3 was selected because it provided meaningful separation while keeping every group large enough for practical operational planning.

The four tier-versus-segment mismatches also show why hub performance should be evaluated using actual operational data rather than existing tier labels alone.

### Segmentation Output Files

- `outputs/hub_profile.csv`
- `outputs/part_d_separation.csv`
- `outputs/hub_segments.csv`
- `outputs/segment_profile.csv`
- `outputs/tier_vs_segment_mismatch.csv`

---

# 6. New Shipment Predictions and Actions (Part D)

## 6.1 Prediction Objective

The final objective was to apply the packaged classification model to 48 new shipments and generate proactive operational actions.

The prediction pipeline:

1. Loads new shipment records.
2. Loads the hub information.
3. Cleans new shipments using the shared preprocessing functions.
4. Uses the saved training-data imputation medians.
5. Builds the same features used during model training.
6. Arranges features in the saved training feature order.
7. Loads the packaged Random Forest model.
8. Calculates the probability of delay.
9. Flags shipments using the saved threshold of 0.5.
10. Joins each shipment with its hub segment.
11. Assigns an operational action.

## 6.2 Alert Threshold

The saved classification threshold is **0.5**.

A shipment is flagged when its predicted probability of delay is greater than or equal to 0.5.

The probability is stored rounded to four decimal places in the output file.

## 6.3 Action Rules

| Condition | Recommended Action |
|---|---|
| Flagged shipment in Escalate segment | Escalate to hub manager |
| Flagged shipment in Standard or Watch segment | Proactive customer call |
| Shipment not flagged | No action |

This approach combines shipment-level delay predictions with hub-level operational information.

## 6.4 New Shipment Results

The prediction pipeline successfully processed all 48 new shipments.

| Result | Number of Shipments |
|---|---:|
| Total shipments | 48 |
| Flagged shipments | 8 |
| Hub manager escalations | 6 |
| Proactive customer calls | 2 |
| No action | 40 |

A total of 8 out of 48 shipments were flagged for potential delays.

Of these, 6 shipments were assigned to hub manager escalation and 2 were assigned to proactive customer calls.

## 6.5 Output File

The final output is saved as:

`outputs/new_shipment_actions.csv`

The file contains the following columns:

| Column | Description |
|---|---|
| shipment_id | Unique shipment identifier |
| hub_id | Associated delivery hub |
| segment | Operational hub segment |
| delay_probability | Predicted probability of delay |
| flagged | 1 if flagged, otherwise 0 |
| action | Recommended operational action |

The output contains 48 rows, one for each new shipment.

Validation checks confirm:

- Exactly 48 shipments were processed.
- Shipment IDs are unique.
- No output values are missing.
- Action assignments follow the defined rules.
- The original shipment order is preserved.

## 6.6 Reproducibility Check

The prediction script was executed twice using the same inputs and saved model.

The SHA256 checksum of the output file was identical in both runs:

```text
a67611c016f731b930962d383a8ab77bc954ed01746646c01ab2833e042a349b
```

This confirms that the two generated CSV files were byte-for-byte identical.

## 6.7 Model Packaging and Reload Validation

The final classifier was saved using Joblib.

The following files were generated:

- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

The packaged model was reloaded and tested against the original model.

| Validation Metric | Result |
|---|---:|
| Test rows scored | 180 |
| Maximum probability difference | 0.0 |
| Saved threshold | 0.5 |

The maximum probability difference was zero, confirming that the reloaded model produced exactly the same probabilities as the original model for the tested rows.

---

# 7. Operations Recommendation

The project provides two complementary levels of operational intelligence:

1. Shipment-level delay prediction.
2. Hub-level operational segmentation.

## 7.1 Hub-Level Recommendations

### Standard Hubs

Continue routine operations and regular performance monitoring.

### Watch Hubs

Monitor delivery performance more frequently. Review route planning, number of stops and operational bottlenecks.

### Escalate Hubs

Prioritize management attention. Investigate capacity constraints, long delivery distances, route inefficiencies and recurring delivery delays.

The four tier-versus-segment mismatches should also receive additional review:

- HUB07 — Kolkata
- HUB13 — Nagpur
- HUB15 — Kochi
- HUB22 — Dehradun

## 7.2 Shipment-Level Recommendations

The model uses a threshold of 0.5.

A flagged shipment in an Escalate hub receives the action:

**Escalate to hub manager**

A flagged shipment in another segment receives:

**Proactive customer call**

A shipment below the threshold receives:

**No action**

## 7.3 Classification Performance Interpretation

On the final test dataset:

- Precision = 0.6889
- Recall = 0.7561

This means approximately 7 out of every 10 flagged shipments were actually delayed in the test set.

The model identified approximately 8 out of every 10 delayed shipments in the test set.

These results should be treated as estimates because the test dataset contained only 41 delayed shipments.

The threshold should be reviewed against actual operational costs, particularly the cost of missed delays versus unnecessary alerts.

## 7.4 New Shipment Recommendations

For the 48 new shipments:

- 8 were flagged.
- 6 require hub manager escalation.
- 2 require proactive customer calls.
- 40 require no immediate action.

These actions provide a starting point for proactive shipment monitoring rather than waiting for a delivery failure to occur.

---

# 8. Production Plan

The current project is an educational prototype. A real production deployment would require additional testing, monitoring and integration with operational systems.

## 8.1 Production Characteristics

| Characteristic | Implementation in This Project | Future Improvement |
|---|---|---|
| Accuracy | Regression and classification metrics are evaluated | Validate on larger real-world datasets |
| Reliability | Model reload checks and prediction assertions are included | Add automated failure alerts |
| Reproducibility | Fixed random states and deterministic predictions | Track data and model versions |
| Performance | Reusable preprocessing functions and saved models | Measure prediction latency on production infrastructure |
| Maintainability | Separate scripts for cleaning, modelling, segmentation and prediction | Add automated tests and structured logging |
| Consistency | Shared preprocessing and saved feature order | Validate incoming data schemas |
| Monitorability | CSV outputs and model evaluation reports | Build dashboards and automated monitoring |

## 8.2 Metrics to Monitor

The following are proposed operational monitoring triggers, not measured production results.

| Metric | Monitoring Frequency | Proposed Trigger |
|---|---|---|
| Weekly actual delay rate | Weekly | Increase of more than 5 percentage points compared with the recent baseline |
| Flagged shipment share | Daily | More than 30% of shipments flagged |
| Classification recall | Monthly, after outcomes are available | Recall below 0.75 |
| Hub delay rate | Weekly | Any hub delay rate above 25% |
| Missing feature values | Daily | More than 2% missing values in required input fields |

When a trigger is reached, the operations or data science team should investigate the cause before changing model settings.

## 8.3 Retraining Schedule

A proposed production schedule would be:

- Monitor model performance weekly.
- Review model performance monthly.
- Retrain quarterly using newly available labelled shipment data.
- Retrain earlier if sustained performance degradation or major operational changes are detected.
- Validate every retrained model before replacing the existing production model.

The test dataset should remain untouched during model development and should not be repeatedly used for threshold selection.

## 8.4 Handling Seasonal and Sale-Event Effects

The synthetic dataset includes a sale-event period from September 15 to September 28.

Sale-event periods can produce different shipment volumes, delivery distances, hub workloads and delay patterns.

Performance during sale periods should therefore be compared with previous periods of similar sale activity rather than only with ordinary weeks.

The model should be monitored separately during:

- Regular operating periods.
- Sale-event periods.
- Major seasonal demand periods.

## 8.5 Data Drift Monitoring

Data drift occurs when the characteristics of incoming shipment data change over time.

Examples include:

- Increasing average delivery distance.
- Changes in parcel weight distribution.
- Increased shipment volume.
- Changes in the proportion of COD shipments.
- Changes in service-type distribution.
- Increased sale-event activity.
- Changes in hub capacity or delivery operations.

The production system should compare incoming feature distributions with the training dataset and investigate significant changes.

---

# 9. Limitations

This project has several limitations.

### 9.1 Synthetic Dataset

The dataset is synthetic and does not represent actual Delhivery operational or customer records. Real-world delivery patterns may be more complex.

### 9.2 Limited Test Data

The classification test dataset contains only 180 shipments, including 41 delayed shipments. Consequently, a small change in the number of correctly identified delayed shipments can noticeably change precision and recall.

### 9.3 Threshold Selection

The classification threshold was selected using a single validation split. Different datasets or operational costs may require a different threshold.

### 9.4 Hub Segmentation

The hub segments are based on only 24 hubs and four operational features. The clusters may change when additional hubs, features or delivery periods are included.

### 9.5 Model Generalisation

Strong performance on synthetic data does not guarantee similar performance on real-world shipment data. The models require validation using actual historical operational data before production use.

### 9.6 Operational Actions

The recommended actions are rule-based suggestions. They do not automatically contact customers or notify hub managers. Real deployment would require integration with operational communication systems and human review.

---

# 10. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates an end-to-end machine learning workflow for delivery intelligence.

The project includes:

- Data cleaning and exploratory analysis.
- Delivery-time prediction using regression.
- Delay classification using Random Forest.
- Feature selection and hyperparameter experiments.
- Classification threshold selection.
- K-means segmentation of 24 delivery hubs.
- Identification of hub tier-versus-performance mismatches.
- Model packaging and reload validation.
- Delay predictions and proactive actions for 48 new shipments.
- Reproducibility checks and production monitoring recommendations.

The project demonstrates how machine learning can support delivery operations by predicting potential delays, identifying underperforming hubs and recommending proactive actions.

**No Waste. Just Intelligence.**
