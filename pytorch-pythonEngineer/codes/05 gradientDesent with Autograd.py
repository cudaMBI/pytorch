import numpy as np
""" 
Linear regression
f = w * x 
f = 2 * x

implementation with everything manual for linear regression:
    1) Prediction ==> Manually.
    2) Gradient Computation ==> Manually
    3) Loss Computation ==> Manually
    4) Parameter updates ==> Manually
    
"""
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ( (y_predicted - y)**2 ).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before the training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # 1) prediction
    y_pred = forward(X)
    
    # 2) loss
    l = loss(Y, y_pred)
    
    # 3) gradients
    dw = gradient(X, Y, y_pred)
    
    # 4) update weights 
    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
 
print(f'Prediction before the training: f(5) = {forward(5):.3f}')        

    
print('\n\n ************************************ \n\n')


""" 
Linear regression
f = w * x 
f = 2 * x

implementation with everything manual except gradient computation for linear regression:
    1) Prediction ==> Manually.
    2) Gradient Computation ==> Autograd
    3) Loss Computation ==> Manually
    4) Parameter updates ==> Manually
    
"""    
import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ( (y_predicted - y)**2 ).mean()

print(f'Prediction before the training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.03
n_iters = 10

for epoch in range(n_iters):
    # 1) prediction
    y_pred = forward(X)
    
    # 2) loss
    l = loss(Y, y_pred)
    
    # 3) gradients == dl/dw
    l.backward()
    
    # 4) update weights 
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # 5) zero gradients
    w.grad.zero_()
    
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
 
print(f'Prediction before the training: f(5) = {forward(5):.3f}')        



