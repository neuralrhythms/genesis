"""
08_exercise_solution.py — Add accident history as a third feature.

IMPROVEMENT OVER 04_normalised.py:
- We add a THIRD input feature: whether the car is accident-free.
- The model is now nn.Linear(3, 1) — three inputs, one output.
- This demonstrates CATEGORICAL ENCODING: converting a text category
  ("None reported" vs "At least 1 accident") into a number (1 or 0).

EXPECTED RESULT: The model should predict a HIGHER price for
accident-free cars compared to cars with reported accidents,
all else being equal.

Run: python 08_exercise_solution.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Data loading and cleaning ─────────────────────────────────────────────────
df = pd.read_csv("./data/used_cars.csv")

age    = df["model_year"].max() - df["model_year"]
milage = df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(int)
price  = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

# ─── New feature: accident-free status ─────────────────────────────────────────
# The "accident" column contains strings like:
#   "None reported"
#   "At least 1 accident or damage reported"
#
# We convert this to a number:
#   1 = no accidents reported (better condition → higher value)
#   0 = at least one accident reported
accident_free = (df["accident"] == "None reported").astype(int)

# ─── Create tensors with 3 features ───────────────────────────────────────────
X = torch.column_stack([
    torch.tensor(accident_free, dtype=torch.float32),  # feature 1: accident-free
    torch.tensor(age,           dtype=torch.float32),  # feature 2: age
    torch.tensor(milage,        dtype=torch.float32),  # feature 3: mileage
])
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# ─── Normalise ─────────────────────────────────────────────────────────────────
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std  = y.std()
y = (y - y_mean) / y_std

# ─── Model: 3 inputs → 1 output ───────────────────────────────────────────────
model     = nn.Linear(3, 1)          # ← 3 inputs now (was 2)
loss_fn   = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# ─── Training ─────────────────────────────────────────────────────────────────
for i in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss    = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 2000 == 0:
        print(f"Step {i:5d} | Loss: {loss.item():.6f}")

# ─── Compare: accident-free vs accident-reported ──────────────────────────────
# Same age (5 years) and mileage (10,000 miles), different accident status
X_clean = torch.tensor([
    [1, 5, 10000],   # accident_free=1
    [1, 2, 10000],
    [1, 5, 20000],
], dtype=torch.float32)

X_accident = torch.tensor([
    [0, 5, 10000],   # accident_free=0
    [0, 2, 10000],
    [0, 5, 20000],
], dtype=torch.float32)

with torch.no_grad():
    pred_clean    = model((X_clean    - X_mean) / X_std) * y_std + y_mean
    pred_accident = model((X_accident - X_mean) / X_std) * y_std + y_mean

print("\n─── Accident-free cars ───")
for i, row in enumerate(X_clean):
    acc, age_val, mil_val = int(row[0]), int(row[1]), int(row[2])
    price_val = pred_clean[i].item()
    print(f"  {age_val}yr old, {mil_val:,} mi → ${price_val:,.0f}")

print("\n─── Cars with accident history ───")
for i, row in enumerate(X_accident):
    acc, age_val, mil_val = int(row[0]), int(row[1]), int(row[2])
    price_val = pred_accident[i].item()
    print(f"  {age_val}yr old, {mil_val:,} mi → ${price_val:,.0f}")

print("\n─── Price difference (clean - accident) ───")
for i in range(len(X_clean)):
    diff = pred_clean[i].item() - pred_accident[i].item()
    print(f"  +${diff:,.0f}")
print("(Positive values confirm accident-free cars are predicted to be worth more)")
