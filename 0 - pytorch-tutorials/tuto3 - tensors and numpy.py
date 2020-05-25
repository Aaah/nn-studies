from __future__ import print_function
import torch
import numpy as np

# -- FROM TENSOR TO NUMPY
a = torch.ones(5); print(a)

# convert to numpy
b = a.numpy(); print(b)

# pointer based : values in b will be updated too
a.add_(1); print(a); print(b)

# -- FROM NUMPY TO TENSOR
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a); print(a); print(b)