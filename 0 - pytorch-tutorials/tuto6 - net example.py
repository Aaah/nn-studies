import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)

        # 6 input channels, 16 outputs channels, 3x3 square conv kernel
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        # 576 inputs features, 120 output features
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        
        # 120 input features, 84 output features
        self.fc2 = nn.Linear(120, 84)

        # 84 input features, 10 output features
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def learnable_parameters(self):
        params = list(net.parameters())
        print(len(params))
        print(params[0].size())  # conv1's .weight
        return

# creation
net = Net()
print(net)
print(net.learnable_parameters())

# -- PROCESSING INPUT DATA
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# 1. torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
# 2. nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
# 3. nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
# 4. autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.

# -- LOSS FUNCTION
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# -- BACKPROPAGATION
net.zero_grad(); print('conv1.bias.grad before backward'); print(net.conv1.bias.grad)
loss.backward(); print('conv1.bias.grad after backward'); print(net.conv1.bias.grad)

# -- OPTIMISATION

optimizer = optim.SGD(net.parameters(), lr=0.01) # create your optimizer

# in the training loop:
optimizer.zero_grad()               # zero the gradient buffers
output = net(input)                 # process data through the network
loss = criterion(output, target)    # compute the loss
loss.backward()                     # backpropagate from the obtained results
optimizer.step()                    # update parameters