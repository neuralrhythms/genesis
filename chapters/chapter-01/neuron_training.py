# This program demonstrates the core idea of neural network training:
# the model starts with random weights, makes predictions, 
# measures error (loss), and iteratively updates its 
# parameters using gradients to minimize that error.

import torch
from torch import nn

# -----------------------------------
# Training Data (Supervised Learning)
# -----------------------------------

# Input: Temperature in °C (feature)
X1 = torch.tensor([[10]], dtype=torch.float32) 
# Actual value: Temperature in °F (label / ground truth)
y1 = torch.tensor([[50]], dtype=torch.float32) 

# Another data point
X2 = torch.tensor([[37.78]], dtype=torch.float32) 
# Corresponding actual value
y2 = torch.tensor([[100.0]], dtype=torch.float32) 

# -----------------------------
# Model Definition (Neuron)
# -----------------------------

# A single neuron: y = wX + b
# 1 input feature → 1 output
model = nn.Linear(1, 1)

# -----------------------------
# Loss Function
# -----------------------------

# Mean Squared Error (MSE)
# Measures how far predictions are from actual values
# Formula: (y_pred - y_actual)^2 averaged over samples
loss_fn = torch.nn.MSELoss()

# -----------------------------
# Optimizer (Learning Mechanism)
# -----------------------------

# Stochastic Gradient Descent (SGD)
# Updates model parameters (weights & bias) to reduce loss
# lr = learning rate → step size of updates
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# -----------------------------
# Training Loop
# -----------------------------

# Repeat training for many iterations (epochs)
for i in range(0, 100000):

    # ---- Train on first data point ----

    # Reset gradients from previous step
    optimizer.zero_grad()

    # Forward pass: compute prediction
    outputs = model(X1)

    # Compute loss (error between prediction and actual value)
    loss = loss_fn(outputs, y1)

    # Backward pass: compute gradients (derivatives of loss w.r.t weights)
    loss.backward()

    # Update weights and bias using gradients
    optimizer.step()

    # ---- Train on second data point ----

    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    # Print parameters periodically to observe learning
    if i % 100 == 0: 
        print(model.bias)
        print(model.weight)

# -----------------------------
# Inference (After Training)
# -----------------------------

# Use trained model to make prediction
y1_pred = model(X1)
y2_pred = model(X2)

# Final prediction should be close to actual values used for training. i.e (y1=50°F, y2=100°F)
print("y1_pred =", y1_pred)
print("y2_pred =", y2_pred)