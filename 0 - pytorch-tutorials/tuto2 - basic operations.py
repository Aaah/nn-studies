from __future__ import print_function
import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)

# syntax 1
z1 = x + y
print(z1)

# syntax 2
z2 = torch.add(x, y)
print(z2)

# syntax 3 - pointer to output
z3 = torch.empty(5,3); torch.add(x, y, out=z3)
print(z3)

# syntax 4 - in place use a _
y.add_(x)

# resizing: if you want to resize/reshape tensor, you can use torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions

print(x.size(), x)
print(y.size(), y)
print(z.size(), z)

