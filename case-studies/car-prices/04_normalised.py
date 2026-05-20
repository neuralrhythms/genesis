"""
04_normalised.py — Train with fully normalised inputs AND outputs.

IMPROVEMENT OVER 03_training_raw.py:
- Both X (inputs) and y (target) are normalised to mean=0, std=1.
- This allows a sensible learning rate (0.001 instead of 0.00000000001).
- Training converges much faster and produces meaningful predictions.
- After prediction, we de-normalise back to real dollar values.

KEY CONCEPT: Normalisation
  normalised_value = (original_value - mean) / standard_deviation
  This puts all features on the same scale so gradient descent works properly.

Run: python 04_normalised.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Data loading and cleaning (same as before) ───────────────────────────────
df = pd.read_csv("./data/used_cars.csv")

age    = df["model_year"].max() - df["model_year"]
milage = df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(int)
price  = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

# ─── Create tensors ────────────────────────────────────────────────────────────
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# ─── Normalise inputs ──────────────────────────────────────────────────────────
# Compute mean and std for each column (age, mileage)
X_mean = X.mean(axis=0)   # shape: (2,) — one mean per feature
X_std  = X.std(axis=0)    # shape: (2,) — one std per feature

# Apply normalisation: each feature now has mean≈0 and std≈1
X = (X - X_mean) / X_std

# ─── Normalise output ──────────────────────────────────────────────────────────
# Same idea for the target (price)
y_mean = y.mean()
y_std  = y.std()
y = (y - y_mean) / y_std

# ─── Model, loss, optimizer ────────────────────────────────────────────────────
model     = nn.Linear(2, 1)
loss_fn   = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # ← sensible LR now!

# ─── Training loop ─────────────────────────────────────────────────────────────
for i in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss    = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"Step {i:5d} | Loss: {loss.item():.6f}")

# ─── Predict on new cars ──────────────────────────────────────────────────────
# IMPORTANT: New data must be normalised using the SAME mean and std
# that were computed from the training data.
X_new = torch.tensor([
    [5, 10000],    # 5-year-old, 10k miles
    [2, 10000],    # 2-year-old, 10k miles
    [5, 20000],    # 5-year-old, 20k miles
], dtype=torch.float32)

# Normalise the new data using training statistics
X_new_normalised = (X_new - X_mean) / X_std

# Predict (output is in normalised scale)
prediction_normalised = model(X_new_normalised)

# De-normalise: convert back to real dollar values
prediction_dollars = prediction_normalised * y_std + y_mean

print("\nPredictions:")
for i, row in enumerate(X_new):
    age_val, mil_val = int(row[0].item()), int(row[1].item())
    price_val = prediction_dollars[i].item()
    print(f"  {age_val}yr old, {mil_val:,} mi → ${price_val:,.0f}")
