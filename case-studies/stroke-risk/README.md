# Case Study 02 — Stroke Risk Prediction

Predict stroke risk using patient symptoms and demographics — first as a regression problem (risk percentage), then as a classification problem (at risk or not).

**Website:** [ai.neuralrhythms.in/case-studies/stroke-risk/](https://ai.neuralrhythms.in/case-studies/stroke-risk/)

---

## Mathematical Foundation

This case study uses **two approaches** on the same dataset, demonstrating the fundamental difference between regression and classification.

### Part A — Regression (predict risk percentage)

Same as Case Study 1: multiple linear regression with MSE loss.

```
risk_percentage = w₁×age + w₂×gender + w₃×chest_pain + ... + b
```

Output: a continuous number between 0 and 100.

### Part B — Classification (predict at-risk yes/no)

A new approach: **logistic regression** — a linear model followed by the **sigmoid activation function**.

```
logit = w₁×age + w₂×gender + w₃×chest_pain + ... + b
probability = sigmoid(logit) = 1 / (1 + e^(-logit))
```

The sigmoid function squashes any number into the range [0, 1], making the output interpretable as a probability.

### Why Sigmoid?

A linear model's raw output is unbounded — it can produce -3 or +7. But a probability must be between 0 and 1. The sigmoid function enforces this constraint:

- When the logit is very negative → sigmoid → ~0 (confident "not at risk")
- When the logit is zero → sigmoid → 0.5 (uncertain)
- When the logit is very positive → sigmoid → ~1 (confident "at risk")

### Why BCE Loss Instead of MSE?

| Loss Function | Designed for | Problem with classification |
|---|---|---|
| MSE (Mean Squared Error) | Regression (continuous outputs) | Treats 0.9→1.0 the same as 0.1→0.2. Doesn't penalise confident wrong predictions enough. |
| BCE (Binary Cross-Entropy) | Classification (probabilities) | Penalises confident wrong answers exponentially. Mathematically correct for binary targets. |

### Evaluation Metrics

Accuracy alone is misleading when classes are imbalanced. The confusion matrix provides:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Sensitivity (Recall) | TP / (TP+FN) | "How many sick people did we catch?" |
| Specificity | TN / (TN+FP) | "How many healthy people did we correctly clear?" |
| Precision | TP / (TP+FP) | "Of those we flagged, how many were actually sick?" |
| F1 Score | 2×(Prec×Sens)/(Prec+Sens) | Balance between precision and sensitivity |

### Feature Importance

Because all symptom inputs are binary (0/1), the learned weights directly indicate each feature's contribution to risk. A large positive weight means that symptom strongly increases predicted stroke risk.

---

## Scripts

Run in order — each builds on the previous:

| # | File | What it does | Key concept |
|---|------|-------------|-------------|
| 01 | `01_explore_data.py` | Load and inspect dataset | Pandas, class balance |
| 02 | `02_regression_risk_score.py` | Predict risk % (regression) | Multiple linear regression, 16 features |
| 03 | `03_classification_mse_fails.py` | Try classification with MSE | Why MSE fails for binary targets |
| 04 | `04_sigmoid_bce.py` | Sigmoid + BCE loss | Activation functions, proper classification |
| 05 | `05_evaluation_metrics.py` | Confusion matrix, F1, threshold | Full model evaluation |
| 06 | `06_train_val_split.py` | Train/validation split | Generalisation, overfitting |
| 07 | `07_feature_importance.py` | Inspect learned weights | Interpretability |
| 08 | `08_predict_new_patients.py` | Predict on new patients | Inference, deployment |

---

## Dataset

**Source:** [Stroke Risk Prediction Dataset v2](https://www.kaggle.com/datasets/mahatiratakandh/stroke-risk-prediction-dataset) by Mahati Ratakondh
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
**Size:** 35,000 rows × 18 columns

### Features (16 inputs)

| Feature | Type | Description |
|---------|------|-------------|
| age | int | Patient age |
| gender | categorical | Male / Female (encoded as 1/0) |
| chest_pain | binary | 0 = absent, 1 = present |
| high_blood_pressure | binary | 0 = absent, 1 = present |
| irregular_heartbeat | binary | 0 = absent, 1 = present |
| shortness_of_breath | binary | 0 = absent, 1 = present |
| fatigue_weakness | binary | 0 = absent, 1 = present |
| dizziness | binary | 0 = absent, 1 = present |
| swelling_edema | binary | 0 = absent, 1 = present |
| neck_jaw_pain | binary | 0 = absent, 1 = present |
| excessive_sweating | binary | 0 = absent, 1 = present |
| persistent_cough | binary | 0 = absent, 1 = present |
| nausea_vomiting | binary | 0 = absent, 1 = present |
| chest_discomfort | binary | 0 = absent, 1 = present |
| cold_hands_feet | binary | 0 = absent, 1 = present |
| snoring_sleep_apnea | binary | 0 = absent, 1 = present |
| anxiety_doom | binary | 0 = absent, 1 = present |

### Targets (2 outputs)

| Target | Type | Description |
|--------|------|-------------|
| stroke_risk_percentage | float (0–100) | Continuous risk score (regression target) |
| at_risk | binary (0/1) | Classification target |

---

## Running the Code

```bash
cd case-studies/stroke-risk
python 01_explore_data.py
python 02_regression_risk_score.py
# ... and so on
```

See [SETUP.md](../../SETUP.md) for full environment setup instructions.

---

## Further Reading

- [Sigmoid Function — Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
- [Logistic Regression — Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [Binary Cross-Entropy — Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression)
- [Confusion Matrix — Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Sensitivity and Specificity — Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- [F1 Score — Wikipedia](https://en.wikipedia.org/wiki/F-score)
- [Feature Scaling — Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)
