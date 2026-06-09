import torch 
from torch import nn

# Define input data as a tensor (multiple data points)
# Each inner list represents one input sample
X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)

# Create a simple linear model (1 input feature → 1 output)
# This is equivalent to a single neuron
model = nn.Linear(1, 1)

# Manually set the bias term of the model
# nn.Parameter tells PyTorch this is a trainable parameter
model.bias = nn.Parameter(
    torch.tensor([32], dtype=torch.float32)
)

# Manually set the weight of the model
# Shape [[1.8]] because PyTorch expects a 2D tensor for weights
model.weight = nn.Parameter(
    torch.tensor([[1.8]], dtype=torch.float32)
)

# Print the model's bias to verify the value
print(model.bias)

# Print the model's weight to verify the value
print(model.weight)

# Pass the input data through the model
# This computes: y = b + w * X for each input value
# Output will be a tensor of predictions
y_pred = model(X)

# Print the predicted outputs
print(y_pred)