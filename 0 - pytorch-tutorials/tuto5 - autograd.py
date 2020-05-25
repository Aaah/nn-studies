from __future__ import print_function
import torch

# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# torch.Tensor is the central class of the package. If you set its attribute .requires_grad as True, it starts to track all operations on it. When you finish your computation you can call .backward() and have all the gradients computed automatically. The gradient for this tensor will be accumulated into .grad attribute.

# To stop a tensor from tracking history, you can call .detach() to detach it from the computation history, and to prevent future computation from being tracked.

# To prevent tracking history (and using memory), you can also wrap the code block in with torch.no_grad():. This can be particularly helpful when evaluating a model because the model may have trainable parameters with requires_grad=True, but for which we don’t need the gradients.

# There’s one more class which is very important for autograd implementation - a Function.

# Tensor and Function are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a .grad_fn attribute that references a Function that has created the Tensor (except for Tensors created by the user - their grad_fn is None).

# If you want to compute the derivatives, you can call .backward() on a Tensor. If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.

# create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True); print(x)

# track an operation
y = x + 2; print(y); print(y.grad_fn)

# do more operations
z = y * y * 3; out = z.mean(); print(z, out)

# computation of the gradient : call backward then get the gradient as x.grad
# out.backward() is equivalent to out.backward(torch.tensor(1.))
out.backward(); print(x.grad)

# -- EXAMPLE VECTOR-JACOBIAN CHAIN RULE
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# now in this case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly, but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
