"""
08_predict_new_patients.py — Predict stroke risk for new, unseen patients.

IMPROVEMENT OVER 07:
- This is the "deployment" script — the model is trained, and we use it
  to assess stroke risk for new patients (not in the dataset).
- Shows both the PROBABILITY (from sigmoid) and the DECISION (at risk or not).
- Demonstrates how a doctor would use this model in practice.

Run: python 08_predict_new_patients.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load, prepare, normalise, train (same setup as 06) ───────────────────────
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

model     = nn.Linear(X_train_norm.shape[1], 1)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(5000):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_norm), y_train)
    loss.backward()
    optimizer.step()

# ─── Define new patients ──────────────────────────────────────────────────────
# Column order: age, gender(M=1/F=0), chest_pain, high_bp, irreg_heartbeat,
#   short_breath, fatigue, dizziness, swelling, neck_jaw, sweating,
#   cough, nausea, chest_discomfort, cold_hands, snoring, anxiety

patients = {
    "Patient A (60yr male, high BP + chest pain + irregular heartbeat)":
        [60, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Patient B (30yr female, no symptoms)":
        [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Patient C (45yr male, fatigue + dizziness + shortness of breath)":
        [45, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Patient D (55yr female, high BP + snoring + anxiety)":
        [55, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    "Patient E (70yr male, chest pain + high BP + fatigue + dizziness + sweating)":
        [70, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
}

# ─── Predict ──────────────────────────────────────────────────────────────────
threshold = 0.5

print("─── Stroke Risk Predictions ───\n")
print(f"{'Patient':<70} {'Probability':>12} {'Decision'}")
print(f"{'─'*70} {'─'*12} {'─'*12}")

model.eval()
with torch.no_grad():
    for description, features in patients.items():
        X_new = torch.tensor([features], dtype=torch.float32)
        X_new_norm = (X_new - X_mean) / X_std

        logit = model(X_new_norm)
        probability = torch.sigmoid(logit).item()
        decision = "⚠️  AT RISK" if probability > threshold else "✓ Low risk"

        print(f"{description:<70} {probability*100:>10.1f}%  {decision}")

print(f"\n(Decision threshold: {threshold*100:.0f}%)")
print("A lower threshold catches more at-risk patients but increases false alarms.")
