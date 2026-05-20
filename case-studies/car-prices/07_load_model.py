"""
07_load_model.py — Load a saved model and make predictions (no training).

IMPROVEMENT OVER 06_save_model.py:
- No training happens here. We simply load the pre-trained weights
  and normalisation stats from disk, then predict on new data.
- This is how you would use a model in production: train once,
  deploy the saved files, and run inference as many times as needed.

PREREQUISITE: Run 06_save_model.py first to generate the ./model/ files.

Run: python 07_load_model.py
"""

import torch
from torch import nn

# ─── Load normalisation statistics ────────────────────────────────────────────
# These were saved during training. We need them to normalise new inputs
# in exactly the same way the training data was normalised.
X_mean = torch.load("./model/X_mean.pt", weights_only=True)
X_std  = torch.load("./model/X_std.pt",  weights_only=True)
y_mean = torch.load("./model/y_mean.pt", weights_only=True)
y_std  = torch.load("./model/y_std.pt",  weights_only=True)

# ─── Recreate the model architecture ──────────────────────────────────────────
# Must match the architecture used during training: 2 inputs → 1 output
model = nn.Linear(2, 1)

# ─── Load the trained weights ─────────────────────────────────────────────────
# load_state_dict fills the model with the saved weights and bias
model.load_state_dict(
    torch.load("./model/model.pt", weights_only=True)
)

# ─── Switch to evaluation mode ────────────────────────────────────────────────
# model.eval() disables training-specific behaviours (like dropout).
# Not strictly necessary for a simple Linear model, but good practice.
model.eval()

# ─── Predict on new cars ──────────────────────────────────────────────────────
X_new = torch.tensor([
    [5, 10000],    # 5-year-old, 10,000 miles
    [2, 10000],    # 2-year-old, 10,000 miles
    [5, 20000],    # 5-year-old, 20,000 miles
], dtype=torch.float32)

# torch.no_grad() tells PyTorch not to track gradients — saves memory
# and is faster. We don't need gradients because we're not training.
with torch.no_grad():
    # Normalise using the saved training statistics
    prediction_normalised = model((X_new - X_mean) / X_std)

    # De-normalise back to real dollar values
    prediction_dollars = prediction_normalised * y_std + y_mean

print("Predictions:")
for i, row in enumerate(X_new):
    age_val, mil_val = int(row[0].item()), int(row[1].item())
    price_val = prediction_dollars[i].item()
    print(f"  {age_val}yr old, {mil_val:,} mi → ${price_val:,.0f}")
