"""
03_training_raw.py — Train the neuron on raw (un-normalised) data.

IMPROVEMENT OVER 02_first_neuron.py:
- We now add a training loop so the model actually LEARNS.
- We define a loss function (MSE) and an optimizer (SGD).
- After training, we make a prediction on a new car.

PROBLEM: The learning rate has to be absurdly small (0.00000000001)
because age ranges 0–50 while mileage ranges 0–200,000. These
mismatched scales make gradient descent unstable. The next script
(04_normalised.py) fixes this with normalisation.

Run: python 03_training_raw.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Data loading and cleaning (same as 02) ───────────────────────────────────
df = pd.read_csv("./data/used_cars.csv")

age = df["model_year"].max() - df["model_year"]

milage = df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(int)
price  = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

# ─── Create tensors ────────────────────────────────────────────────────────────
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

# Target: price reshaped to (4009, 1) so it matches the model output shape
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# ─── Model, loss, optimizer ────────────────────────────────────────────────────
model     = nn.Linear(2, 1)          # 2 inputs → 1 output
loss_fn   = torch.nn.MSELoss()       # Mean Squared Error: average of (pred - actual)²
optimizer = torch.optim.SGD(         # Stochastic Gradient Descent
    model.parameters(),
    lr=0.00000000001                 # ← absurdly small LR needed because data is not normalised
)

# ─── Training loop ─────────────────────────────────────────────────────────────
for i in range(1000):
    optimizer.zero_grad()            # Reset gradients from previous step
    outputs = model(X)               # Forward pass: compute predictions
    loss    = loss_fn(outputs, y)    # Compute how wrong the predictions are
    loss.backward()                  # Backpropagate: compute gradients
    optimizer.step()                 # Update weights using gradients

    if i % 100 == 0:
        print(f"Step {i:4d} | Loss: {loss.item():.2f}")

# ─── Predict on a new car ─────────────────────────────────────────────────────
# 5-year-old car with 20,000 miles
new_car = torch.tensor([[5, 20000]], dtype=torch.float32)
prediction = model(new_car)
print(f"\nPredicted price (5yr, 20k mi): ${prediction.item():,.0f}")
print("(This prediction may be poor because training without normalisation is unstable)")
