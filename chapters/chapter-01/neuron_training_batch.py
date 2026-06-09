# This version of the progra introduces batch training, 
# where multiple data points are processed together. 
# The model learns from the average error across the batch, 
# making training more efficient and stable 
# compared to updating on one sample at a time.


import torch
from torch import nn

# -----------------------------------
# Training Data (Batch Learning)
# -----------------------------------

# Inputs: Temperatures in °C (multiple samples at once)
# Shape: (batch_size, features) → here (2, 1)
X = torch.tensor([
    [10],
    [37.78]
], dtype=torch.float32)

# Targets: Corresponding temperatures in °F
# Same batch size as inputs
# Shape: (2, 1)
y = torch.tensor([
    [50],
    [100.0]
], dtype=torch.float32)

# -----------------------------
# Model Definition
# -----------------------------

# Single neuron (linear layer)
# Learns mapping: y = wX + b
model = nn.Linear(1, 1)

# -----------------------------
# Loss Function
# -----------------------------

# Mean Squared Error across the entire batch
# Computes average error over all samples in the batch
loss_fn = torch.nn.MSELoss()

# -----------------------------
# Optimizer
# -----------------------------

# Gradient Descent optimizer
# Updates weights using average gradient from the batch
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# -----------------------------
# Training Loop
# -----------------------------

for i in range(0, 150000):

    # Reset gradients (important before each step)
    optimizer.zero_grad()

    # Forward pass on entire batch
    # Model processes all inputs at once (vectorized computation)
    outputs = model(X)

    # Compute loss across the batch
    # Loss is averaged over all samples
    loss = loss_fn(outputs, y)

    # Backward pass
    # Computes gradients of loss w.r.t model parameters
    loss.backward()

    # Update weights and bias
    optimizer.step()

    # Monitor learning progress
    if i % 100 == 0: 
        print(model.bias)
        print(model.weight)

print("----")

# -----------------------------
# Inference (Evaluation Mode)
# -----------------------------

# New unseen input (not part of training data)
measurements = torch.tensor([
    [37.5]
], dtype=torch.float32)

# Switch model to evaluation mode
# Disables behaviors like dropout (not used here but good practice)
model.eval()

# Disable gradient tracking (faster, memory efficient)
with torch.no_grad():

    # Make prediction
    prediction = model(measurements)

    # Output predicted Fahrenheit value
    print(prediction)