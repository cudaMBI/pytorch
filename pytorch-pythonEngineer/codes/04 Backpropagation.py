
import torch

"""
1 - Forward pass: compute loss
2 - At each node: compute local gradients
3 - Backward pass: compute dLoss/dWeights using the chain rule 
"""

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# 1) forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss.item())

# 2) pytorch will automatically compute the loacal gradients and backward pass
loss.backward() 
print(w.grad) 

# 3) update our weights, and the next forward and backward pass for a number of iterations