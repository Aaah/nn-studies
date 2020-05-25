from __future__ import print_function
import torch

# create a matrix with defaults values are defined by what was in the memory
x = torch.empty(5, 3)
print(x)

# construct a randomly initialized matrix with values between 0 and 1
x = torch.rand(5, 3)
print(x)

# construct a matrix with zeros, data type is long (32bits floats)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# replace existing tensors, converting types
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

# get the size of a tensor
print(x.size())