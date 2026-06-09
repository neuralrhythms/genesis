"""
06_train_val_split.py — Split data into training and validation sets.

IMPROVEMENT OVER 05:
- Previously we trained AND evaluated on the SAME data. This is cheating!
  The model memorises the training data and looks perfect — but may fail
  on new, unseen patients.
- Now we split: 80% for training, 20% for validation (unseen data).
- We train on the training set and evaluate on BOTH sets.
- If training metrics are great but validation metrics are poor, the model
  is OVERFITTING — memorising instead of generalising.

KEY CONCEPT: GENERALISATION
  A good model performs well on data it has NEVER seen before.
  The validation set simulates "new patients walking into the hospital."

Run: python 06_train_val_split.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Load and prepare ──────────────────────────────────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")
df["gender"] = (df["gender"] == "Male").astype(int)

feature_columns = [col for col in df.columns if col not in ["stroke_risk_percentage", "at_risk"]]

# ─── Split: 80% train, 20% validation ─────────────────────────────────────────
# random_state=42 ensures the split is reproducible (same split every run)
df_train = df.sample(frac=0.8, random_state=42)
df_val   = df.drop(index=df_train.index)

print(f"Training set:   {len(df_train)} patients")
print(f"Validation set: {len(df_val)} patients")

# ─── Create tensors ────────────────────────────────────────────────────────────
X_train = torch.tensor(df_train[feature_columns].values, dtype=torch.float32)
y_train = torch.tensor(df_train["at_risk"].values, dtype=torch.float32).reshape((-1, 1))

X_val = torch.tensor(df_val[feature_columns].values, dtype=torch.float32)
y_val = torch.tensor(df_val["at_risk"].values, dtype=torch.float32).reshape((-1, 1))

# ─── Normalise using TRAINING statistics only ──────────────────────────────────
# IMPORTANT: Compute mean/std from training data ONLY.
# The validation set must be normalised using training statistics —
# because in production, you won't know the statistics of future data.
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_std[X_std == 0] = 1.0

X_train_norm = (X_train - X_mean) / X_std
X_val_norm   = (X_val   - X_mean) / X_std    # ← Same mean/std from training!

# ─── Train ─────────────────────────────────────────────────────────────────────
model     = nn.Linear(X_train_norm.shape[1], 1)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(5000):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_norm), y_train)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        # Also compute validation loss (without training on it!)
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_norm), y_val)
        print(f"Step {i:5d} | Train loss: {loss.item():.6f} | Val loss: {val_loss.item():.6f}")

# ─── Evaluate on both sets ─────────────────────────────────────────────────────
def evaluate(model, X, y, label, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        y_pred = (probabilities > threshold).float()

    TP = ((y_pred == 1) & (y == 1)).sum().item()
    TN = ((y_pred == 0) & (y == 0)).sum().item()
    FP = ((y_pred == 1) & (y == 0)).sum().item()
    FN = ((y_pred == 0) & (y == 1)).sum().item()

    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\n─── {label} (threshold={threshold}) ───")
    print(f"  Accuracy:    {accuracy*100:.1f}%")
    print(f"  Sensitivity: {sensitivity*100:.1f}%")
    print(f"  Precision:   {precision*100:.1f}%")
    print(f"  F1 Score:    {f1*100:.1f}%")

print()
evaluate(model, X_train_norm, y_train, "Training Set")
evaluate(model, X_val_norm,   y_val,   "Validation Set")

print("\n─── What to look for ───")
print("If training metrics >> validation metrics → model is overfitting.")
print("If both are similar → model generalises well.")
