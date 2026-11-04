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