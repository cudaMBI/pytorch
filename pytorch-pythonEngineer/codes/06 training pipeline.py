# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn

""" 
Linear regression
f = w * x 
f = 2 * x

implementation with everything automatically except prediction for linear regression:
    1) Prediction ==> Manually.
    2) Gradient Computation ==> Autograd
    3) Loss Computation ==> pytorch loss
    4) Parameter updates ==> pytorch optimizer
    
"""    

# 0) Training samples
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# 1) Design Model: Weights to optimize and forward function
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 50

# callable function
loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr=learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # 1) prediction
    y_pred = forward(X)

    # 2) loss
    l = loss(Y, y_pred)
    

    # 3) gradients == dl/dw
    l.backward()

    # 4) update weights 
    optimizer.step()

    # # 5) zero gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')


print('\n\n ************************************ \n\n')

##############################################################################
##############################################################################

# we can use models in pytorch instead of implementing the forward function

# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn

""" 
Linear regression
f = w * x 
f = 2 * x

implementation with everything automatically for linear regression:
    1) Prediction ==> pythorch model.
    2) Gradient Computation ==> Autograd
    3) Loss Computation ==> pytorch loss
    4) Parameter updates ==> pytorch optimizer
    
""" 

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')
# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Design Model, the model has to implement the forward pass!
# Here we can use a built-in model from PyTorch
input_size = n_features
output_size = n_features

# we can call this model with samples X
model = nn.Linear(input_size, output_size)

'''
# custom linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_size)
'''

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # 1) predict = forward pass with our model
    y_pred = model(X)

    # 2) loss
    l = loss(Y, y_pred)
    

    # 3) gradients == dl/dw
    l.backward()

    # 4) update weights 
    optimizer.step()

    # # 5) zero gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

