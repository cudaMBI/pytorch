# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:35:21 2022
https://www.youtube.com/watch?v=c36lUUr864M&ab_channel=PythonEngineer

@author: Mo_Badr
"""

import torch

x = torch.empty(1)
print(x)

# 1d vector with 3 elements
x = torch.empty(3)      
print(x)
print(x.size())

# between 0 and 1
x = torch.rand((2, 2))
print(x)

x = torch.zeros(3, 2 , dtype=torch.int)
print(x)
print(x.size())     # torch.Size([3, 2])
print(x.dtype)      # torch.int32

x = torch.zeros(3, 2 , dtype=torch.double)
print(x)
print(x.dtype)      # torch.float64

# 4d 
x = torch.ones(1, 4, 2, 2)
print(x)
print(x.size())     # torch.Size([1, 4, 2, 2])
print(x.dtype)      # torch.float32


x = torch.tensor([2.5, 0, 5.5])
print(x)
print(x.dtype)      # torch.float32

# element-wise addition
x = torch.rand((2, 2))
y = torch.rand((2, 2))
print(x)
print(y)
print(x + y)
print(torch.add(x, y))
print(torch.sub(x, y))
print(torch.mul(x, y))
print(torch.div(x, y))


# inplace addition
x = torch.rand((2, 2))
y = torch.rand((2, 2))
print(x)
print(y)
y.add_(x)
y.sub_(x)
y.mul_(x)
y.div_(x)
print(y)

# slicing
x = torch.rand((5, 4))
print(x[:, :2])
print(x[1, 1])          # tensor(0.0646)
print(x[1, 1].item())   # get actual value = 0.064635 for only one element

# reshaping
x = torch.rand((4, 4))
print(x)
y = x.view(16)      # 1d vector 
print(y)

x = torch.rand((4, 4))
print(x)
y = x.view(-1, 8)    # 2 * 8  
print(y)

### converting from numpy to torch tensor and vice versa
import numpy as np

# from torch to numpy
x = torch.ones(5)
print(x)        # tensor([1., 1., 1., 1., 1.])
y = x.numpy()       
print(y)        # [1. 1. 1. 1. 1.]
print(type(y))  # <class 'numpy.ndarray'>

# note x and y share the same memory location
x.add_(5)
print(x)        # tensor([6., 6., 6., 6., 6.])
print(y)        # [6. 6. 6. 6. 6.]

y -= 2
print(x)        # tensor([4., 4., 4., 4., 4.])    
print(y)        # [4. 4. 4. 4. 4.]

# from numpy to torch
x = np.ones(5)
y = torch.from_numpy(x)
print(x)
print(y)

# note x and y share the same memory location
x *= 3
print(x)
print(y)

# if you have a variable you need to optimize
x = torch.ones(5, requires_grad=True)
print(x)

