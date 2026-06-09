"""
05_evaluation_metrics.py — Evaluate the classifier with proper metrics.

IMPROVEMENT OVER 04:
- Accuracy alone is misleading when classes are imbalanced.
- We compute the full confusion matrix and derive:
  - Accuracy: overall correct predictions / total
  - Sensitivity (Recall): correctly identified at-risk / total actually at-risk
  - Specificity: correctly identified not-at-risk / total actually not-at-risk
  - Precision: correctly identified at-risk / total predicted at-risk
  - F1 Score: harmonic mean of precision and sensitivity
- We also show how THRESHOLD TUNING affects these metrics.

THE CONFUSION MATRIX:
                          Predicted Positive    Predicted Negative
  Actually Positive       True Positive (TP)    False Negative (FN)
  Actually Negative       False Positive (FP)   True Negative (TN)

  Accuracy    = (TP + TN) / (TP + TN + FP + FN)
  Sensitivity = TP / (TP + FN)      "Of all sick people, how many did we catch?"
  Specificity = TN / (TN + FP)      "Of all healthy people, how many did we correctly clear?"
  Precision   = TP / (TP + FP)      "Of all people we flagged, how many were actually sick?"
  F1 Score    = 2 × (Precision × Sensitivity) / (Precision + Sensitivity)

Run: python 05_evaluation_metrics.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load, prepare, normalise ──────────────────────────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")
df["gender"] = (df["gender"] == "Male").astype(int)

feature_columns = [col for col in df.columns if col not in ["stroke_risk_percentage", "at_risk"]]
X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
y = torch.tensor(df["at_risk"].values, dtype=torch.float32).reshape((-1, 1))

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# ─── Train ─────────────────────────────────────────────────────────────────────
model     = nn.Linear(X_norm.shape[1], 1)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(5000):
    optimizer.zero_grad()
    loss = loss_fn(model(X_norm), y)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.6f}\n")

# ─── Evaluation function ──────────────────────────────────────────────────────
def evaluate(model, X, y, threshold=0.5):
    """Compute and print all classification metrics."""
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        y_pred = (probabilities > threshold).float()

    # Confusion matrix components
    TP = ((y_pred == 1) & (y == 1)).sum().item()
    TN = ((y_pred == 0) & (y == 0)).sum().item()
    FP = ((y_pred == 1) & (y == 0)).sum().item()
    FN = ((y_pred == 0) & (y == 1)).sum().item()

    # Metrics
    accuracy    = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"  Threshold:   {threshold}")
    print(f"  Accuracy:    {accuracy*100:.1f}%")
    print(f"  Sensitivity: {sensitivity*100:.1f}%  (recall — catches sick patients)")
    print(f"  Specificity: {specificity*100:.1f}%  (correctly clears healthy patients)")
    print(f"  Precision:   {precision*100:.1f}%  (flagged patients who are actually sick)")
    print(f"  F1 Score:    {f1*100:.1f}%")
    print(f"  TP={int(TP)}, TN={int(TN)}, FP={int(FP)}, FN={int(FN)}")
    print()

# ─── Evaluate at different thresholds ──────────────────────────────────────────
print("─── Evaluation at threshold = 0.5 ───")
evaluate(model, X_norm, y, threshold=0.5)

print("─── Evaluation at threshold = 0.3 ───")
evaluate(model, X_norm, y, threshold=0.3)

print("─── Evaluation at threshold = 0.7 ───")
evaluate(model, X_norm, y, threshold=0.7)

# ─── Insight ──────────────────────────────────────────────────────────────────
print("─── Threshold trade-offs ───")
print("Lower threshold (0.3): catches more at-risk patients (higher sensitivity)")
print("                       but also flags more healthy patients (lower specificity)")
print("Higher threshold (0.7): fewer false alarms (higher precision)")
print("                        but misses more at-risk patients (lower sensitivity)")
print("\nIn medicine, SENSITIVITY is usually prioritised — missing a stroke is worse")
print("than a false alarm.")
