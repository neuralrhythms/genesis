"""
02_first_neuron.py — Create a multi-feature neuron (no training yet).

This script shows how to:
1. Clean the raw string data (mileage, price) into numbers
2. Stack two features (age, mileage) into a single input tensor
3. Create a neuron with 2 inputs and 1 output
4. Make a prediction (with random, untrained weights)

This is the equivalent of Chapter 1's single neuron, but with TWO inputs
instead of one. The equation is now: price = w1*age + w2*mileage + b

Run: python 02_first_neuron.py
"""

import pandas as pd
import torch
from torch import nn

# ─── Step 1: Load the data ─────────────────────────────────────────────────────
df = pd.read_csv("./data/used_cars.csv")

# ─── Step 2: Clean and prepare features ────────────────────────────────────────

# Derive "age" from model_year.
# A 2024 car has age 0, a 2014 car has age 10.
# This is more meaningful than the raw year number.
age = df["model_year"].max() - df["model_year"]

# Mileage is stored as a string like "51,000 mi."
# We strip the commas and the " mi." suffix, then convert to integer.
milage = df["milage"]
milage = milage.str.replace(",", "")       # "51,000 mi." → "51000 mi."
milage = milage.str.replace(" mi.", "")    # "51000 mi."  → "51000"
milage = milage.astype(int)                # "51000"      → 51000

# Price is stored as "$10,300" — strip $ and commas, convert to int.
price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

# ─── Step 3: Create PyTorch tensors ────────────────────────────────────────────

# Stack age and mileage side by side into a (4009, 2) tensor.
# Each row is one car: [age, mileage]
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

# ─── Step 4: Create the neuron ─────────────────────────────────────────────────

# nn.Linear(2, 1) means: 2 inputs → 1 output
# Internally this computes: output = w1*input1 + w2*input2 + bias
# The weights start random — the model hasn't learned anything yet.
model = nn.Linear(2, 1)

# ─── Step 5: Make a prediction (untrained) ─────────────────────────────────────

# Pass ALL 4009 cars through the model at once.
# The output will be random nonsense because the weights are random.
prediction = model(X)
print("First 5 predictions (untrained, random weights):")
print(prediction[:5])
