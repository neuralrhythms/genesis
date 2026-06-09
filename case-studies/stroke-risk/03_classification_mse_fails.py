"""
03_classification_mse_fails.py — Try classification with MSE loss (it fails).

IMPROVEMENT OVER 02:
- Instead of predicting risk percentage, we now predict at_risk (0 or 1).
- We use the SAME approach as before: linear neuron + MSE loss.
- PROBLEM: The output is not bounded between 0 and 1. The model outputs
  numbers like -0.3 or 1.4 — which don't make sense as probabilities.

This script INTENTIONALLY shows a broken approach. The next script (04)
fixes it by introducing the sigmoid activation function.

Run: python 03_classification_mse_fails.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load and prepare data ─────────────────────────────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")
df["gender"] = (df["gender"] == "Male").astype(int)

feature_columns = [col for col in df.columns if col not in ["stroke_risk_percentage", "at_risk"]]
X = torch.tensor(df[feature_columns].values, dtype=torch.float32)

# Target: at_risk (binary: 0 or 1)
y = torch.tensor(df["at_risk"].values, dtype=torch.float32).reshape((-1, 1))

# ─── Normalise inputs only (target is already 0/1) ────────────────────────────
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# ─── Model: 16 inputs → 1 output, MSE loss (wrong for classification!) ────────
model     = nn.Linear(X_norm.shape[1], 1)
loss_fn   = nn.MSELoss()           # ← This is designed for regression, NOT classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ─── Training ─────────────────────────────────────────────────────────────────
for i in range(5000):
    optimizer.zero_grad()
    outputs = model(X_norm)
    loss    = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"Step {i:5d} | Loss: {loss.item():.6f}")

# ─── Inspect raw output ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = model(X_norm)

print(f"\nRaw output statistics:")
print(f"  Min:  {y_pred.min().item():.4f}")    # ← Can go below 0!
print(f"  Max:  {y_pred.max().item():.4f}")    # ← Can go above 1!
print(f"  Mean: {y_pred.mean().item():.4f}")

# ─── The problem ──────────────────────────────────────────────────────────────
# The model's output is unbounded. It can produce -0.3 or 1.4.
# But a probability MUST be between 0 and 1.
# We need something that squashes any number into the 0–1 range.
# That something is the SIGMOID function → see 04_sigmoid_bce.py

below_zero = (y_pred < 0).sum().item()
above_one  = (y_pred > 1).sum().item()
print(f"\n  Predictions below 0: {below_zero}")
print(f"  Predictions above 1: {above_one}")
print(f"\n→ MSE loss does not constrain outputs to valid probabilities.")
print(f"→ The fix: SIGMOID activation function + BCE loss (see next script).")
