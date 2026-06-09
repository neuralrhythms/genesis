"""
07_feature_importance.py — Which symptoms matter most for stroke risk?

IMPROVEMENT OVER 06:
- After training, we inspect the model's learned WEIGHTS.
- Each weight corresponds to one input feature.
- A large positive weight means that feature INCREASES stroke risk.
- A large negative weight means it DECREASES risk (or is protective).
- A weight near zero means the feature barely matters.

WHY THIS WORKS:
  Because all symptom features are binary (0 or 1), the weight directly
  tells you how much that symptom contributes to the prediction.
  This is a unique advantage of this dataset — the model is directly
  interpretable without any extra tools.

Run: python 07_feature_importance.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load, prepare, split, normalise (same as 06) ─────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")
df["gender"] = (df["gender"] == "Male").astype(int)

feature_columns = [col for col in df.columns if col not in ["stroke_risk_percentage", "at_risk"]]

df_train = df.sample(frac=0.8, random_state=42)
X_train = torch.tensor(df_train[feature_columns].values, dtype=torch.float32)
y_train = torch.tensor(df_train["at_risk"].values, dtype=torch.float32).reshape((-1, 1))

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_std[X_std == 0] = 1.0
X_train_norm = (X_train - X_mean) / X_std

# ─── Train ─────────────────────────────────────────────────────────────────────
model     = nn.Linear(X_train_norm.shape[1], 1)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(5000):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_norm), y_train)
    loss.backward()
    optimizer.step()

# ─── Extract and rank weights ──────────────────────────────────────────────────
weights = model.weight.data.squeeze()  # Shape: (16,)
bias    = model.bias.data.item()

# Pair each feature name with its weight
feature_weights = list(zip(feature_columns, weights.tolist()))

# Sort by absolute weight (most important first)
feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

print("─── Feature Importance (sorted by |weight|) ───\n")
print(f"{'Feature':<25} {'Weight':>8}  {'Direction'}")
print(f"{'─'*25} {'─'*8}  {'─'*20}")

for name, w in feature_weights:
    direction = "↑ increases risk" if w > 0 else "↓ decreases risk"
    print(f"{name:<25} {w:>8.4f}  {direction}")

print(f"\n{'Bias':<25} {bias:>8.4f}")

print("\n─── Interpretation ───")
print("Larger |weight| = stronger influence on prediction.")
print("Positive weight = symptom INCREASES stroke risk.")
print("Negative weight = symptom DECREASES stroke risk (or is protective).")
print("\nThe top 3-5 features are the strongest predictors of stroke in this model.")
