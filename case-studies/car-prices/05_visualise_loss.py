"""
05_visualise_loss.py — Plot the loss curve to confirm the model is learning.

IMPROVEMENT OVER 04_normalised.py:
- We record the loss at every step and plot it with Matplotlib.
- A healthy loss curve falls steeply at first, then flattens.
- If the curve is flat → learning rate too small.
- If the curve spikes/explodes → learning rate too large.

Run: python 05_visualise_loss.py
"""

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# ─── Data loading, cleaning, normalisation (same as 04) ───────────────────────
df = pd.read_csv("./data/used_cars.csv")

age    = df["model_year"].max() - df["model_year"]
milage = df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(int)
price  = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
y_mean = y.mean()
y_std  = y.std()
y = (y - y_mean) / y_std

# ─── Model setup ──────────────────────────────────────────────────────────────
model     = nn.Linear(2, 1)
loss_fn   = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ─── Training loop — record losses ────────────────────────────────────────────
losses = []                          # We'll store the loss value at each step

for i in range(250):
    optimizer.zero_grad()
    outputs = model(X)
    loss    = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    # .item() extracts the Python float from the tensor
    losses.append(loss.item())

# ─── Plot the loss curve ──────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(losses, color="#4a6fa5", linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Loss (MSE, normalised)")
plt.title("Loss Curve — Car Price Model")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ─── Final predictions ────────────────────────────────────────────────────────
X_new = torch.tensor([
    [5, 10000],
    [2, 10000],
    [5, 20000],
], dtype=torch.float32)

prediction = model((X_new - X_mean) / X_std)
print("\nPredictions (after 250 steps):")
print(prediction * y_std + y_mean)
