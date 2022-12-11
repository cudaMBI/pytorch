# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:36:20 2022

@author: Mo_Badr
"""

import torch

# we want to calculate the gradiant of some function with respect to x
x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y+2
z = z.mean()
print(z)

# calculate the gradiant
z.backward()  # dz/dx
print(x.grad)


##  3 ways to stop torch from calculating gradiant
x = torch.rand(3, requires_grad=True)
print(x)

#x.requires_grad_(False)
#print(x)

#y = x.detach()
#print(y)

with torch.no_grad():
    y = x+2
    print(y)
    
#####
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights+3).sum()
    
    model_output.backward()
    
    print(weights.grad)
    
    weights.grad.zero_()
    