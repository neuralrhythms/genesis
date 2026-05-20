"""
06_save_model.py — Train the model and save everything to disk.

IMPROVEMENT OVER 05_visualise_loss.py:
- After training, we save the model weights AND the normalisation
  statistics (X_mean, X_std, y_mean, y_std) to a ./model/ folder.
- This means we can load the model later and make predictions
  WITHOUT retraining (see 07_load_model.py).

WHY save normalisation stats?
  New input data must be normalised using the SAME mean and std
  that were computed during training. If you lose these numbers,
  the model's predictions will be meaningless.

Run: python 06_save_model.py
"""

import os
import pandas as pd
import torch
from torch import nn

# ─── Data loading, cleaning ───────────────────────────────────────────────────
df = pd.read_csv("./data/used_cars.csv")

age    = df["model_year"].max() - df["model_year"]
milage = df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(int)
price  = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

# ─── Create output directory ──────────────────────────────────────────────────
# exist_ok=True means: don't error if the folder already exists
os.makedirs("./model", exist_ok=True)

# ─── Create and normalise tensors ─────────────────────────────────────────────
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)

# Save normalisation stats BEFORE applying them
torch.save(X_mean, "./model/X_mean.pt")
torch.save(X_std,  "./model/X_std.pt")

X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
y_mean = y.mean()
y_std  = y.std()

torch.save(y_mean, "./model/y_mean.pt")
torch.save(y_std,  "./model/y_std.pt")

y = (y - y_mean) / y_std

# ─── Train ─────────────────────────────────────────────────────────────────────
model     = nn.Linear(2, 1)
loss_fn   = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(2500):
    optimizer.zero_grad()
    outputs = model(X)
    loss    = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.6f}")

# ─── Save model weights ───────────────────────────────────────────────────────
# state_dict() contains all learned parameters (weights and bias)
torch.save(model.state_dict(), "./model/model.pt")

print("Model and normalisation stats saved to ./model/")
print("Files: model.pt, X_mean.pt, X_std.pt, y_mean.pt, y_std.pt")
