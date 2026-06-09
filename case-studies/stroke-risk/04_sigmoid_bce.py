"""
04_sigmoid_bce.py — Proper classification with sigmoid + BCE loss.

IMPROVEMENT OVER 03:
- Replaces MSE with BCEWithLogitsLoss — the correct loss for binary classification.
- BCEWithLogitsLoss internally applies the SIGMOID function, squashing the
  model's raw output into a valid probability between 0 and 1.
- The model now outputs probabilities: "72% chance this patient is at risk."

THE SIGMOID FUNCTION:
  sigmoid(x) = 1 / (1 + e^(-x))

  - Input: any number from -∞ to +∞
  - Output: always between 0 and 1
  - When x=0: sigmoid(0) = 0.5 (perfect uncertainty)
  - When x is very positive: sigmoid → 1 (confident "at risk")
  - When x is very negative: sigmoid → 0 (confident "not at risk")

WHY BCE (Binary Cross-Entropy)?
  MSE treats the difference between 0.9 and 1.0 the same as 0.1 and 0.2.
  BCE is designed for probabilities — it penalises confident wrong answers
  much more heavily than uncertain ones. This makes training faster and
  more stable for classification.

Run: python 04_sigmoid_bce.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load and prepare ──────────────────────────────────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")
df["gender"] = (df["gender"] == "Male").astype(int)

feature_columns = [col for col in df.columns if col not in ["stroke_risk_percentage", "at_risk"]]
X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
y = torch.tensor(df["at_risk"].values, dtype=torch.float32).reshape((-1, 1))

# ─── Normalise inputs ──────────────────────────────────────────────────────────
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# ─── Model + BCE loss ──────────────────────────────────────────────────────────
model     = nn.Linear(X_norm.shape[1], 1)
loss_fn   = nn.BCEWithLogitsLoss()   # ← Sigmoid + BCE combined (numerically stable)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ─── Training ─────────────────────────────────────────────────────────────────
for i in range(5000):
    optimizer.zero_grad()
    outputs = model(X_norm)          # Raw logits (not yet probabilities)
    loss    = loss_fn(outputs, y)    # BCE internally applies sigmoid
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"Step {i:5d} | Loss: {loss.item():.6f}")

# ─── Predict probabilities ────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    # Apply sigmoid manually to get probabilities from raw logits
    logits       = model(X_norm)
    probabilities = torch.sigmoid(logits)

print(f"\nProbability statistics:")
print(f"  Min:  {probabilities.min().item():.4f}")    # ← Always ≥ 0
print(f"  Max:  {probabilities.max().item():.4f}")    # ← Always ≤ 1
print(f"  Mean: {probabilities.mean().item():.4f}")

# ─── Convert probabilities to predictions ──────────────────────────────────────
# A probability above 0.5 = "at risk", below 0.5 = "not at risk"
threshold = 0.5
y_pred = (probabilities > threshold).float()

# ─── Quick accuracy check ─────────────────────────────────────────────────────
accuracy = (y_pred == y).float().mean()
print(f"\nAccuracy (threshold={threshold}): {accuracy.item()*100:.1f}%")
print(f"(Full evaluation metrics in the next script)")
