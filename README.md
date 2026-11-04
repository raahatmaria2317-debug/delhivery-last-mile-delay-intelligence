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