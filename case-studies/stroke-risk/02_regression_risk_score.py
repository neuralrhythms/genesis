"""
02_regression_risk_score.py — Predict stroke risk percentage (regression).

This script uses the SAME approach as Case Study 1 (Car Prices):
- A linear neuron (nn.Linear)
- MSE loss
- Normalisation of inputs and output
- Gradient descent

The only difference: we now have 16 input features instead of 2.
The model learns: risk = w₁×age + w₂×gender + w₃×chest_pain + ... + b

This is multiple linear regression — same as Case Study 1, just bigger.

Run: python 02_regression_risk_score.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load and prepare data ─────────────────────────────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")

# Encode gender: Male=1, Female=0
df["gender"] = (df["gender"] == "Male").astype(int)

# ─── Create tensors ────────────────────────────────────────────────────────────
# All 17 input features: age, gender, and 15 symptom columns
feature_columns = [col for col in df.columns if col not in ["stroke_risk_percentage", "at_risk"]]
X = torch.tensor(df[feature_columns].values, dtype=torch.float32)

# Target: stroke risk percentage (continuous, 0–100)
y = torch.tensor(df["stroke_risk_percentage"].values, dtype=torch.float32).reshape((-1, 1))

print(f"Input shape:  {X.shape}")   # → (35000, 17)
print(f"Target shape: {y.shape}")   # → (35000, 1)

# ─── Normalise ─────────────────────────────────────────────────────────────────
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
# Avoid division by zero for columns with zero std (unlikely here but safe)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std  = y.std()
y_norm = (y - y_mean) / y_std

# ─── Model: 17 inputs → 1 output (regression) ─────────────────────────────────
num_features = X.shape[1]  # 17
model     = nn.Linear(num_features, 1)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ─── Training loop ─────────────────────────────────────────────────────────────
for i in range(5000):
    optimizer.zero_grad()
    outputs = model(X_norm)
    loss    = loss_fn(outputs, y_norm)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"Step {i:5d} | Loss: {loss.item():.6f}")

# ─── Predict on sample patients ───────────────────────────────────────────────
# Patient A: 55-year-old male with high blood pressure and chest pain
# Patient B: 25-year-old female with no symptoms
# Columns: age, gender, chest_pain, high_bp, irreg_hb, short_breath, fatigue,
#           dizziness, swelling, neck_jaw, sweating, cough, nausea, chest_disc,
#           cold_hands, snoring, anxiety
patient_A = torch.tensor([[55, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
patient_B = torch.tensor([[25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)

with torch.no_grad():
    pred_A = model((patient_A - X_mean) / X_std) * y_std + y_mean
    pred_B = model((patient_B - X_mean) / X_std) * y_std + y_mean

print(f"\nPatient A (55yr male, chest pain + high BP): {pred_A.item():.1f}% risk")
print(f"Patient B (25yr female, no symptoms):        {pred_B.item():.1f}% risk")
