import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

do_training = True
MODEL_PATH = './cifar_net.pth'

# the output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# the dataset is downloaded if needed and the "transform" is applied
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

print(trainset)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# the data loader will ouput images 4 by 4, shuffled, using 2 subprocesses to speed things up
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# classes we'll want to train the network to discriminate
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # -- SHOW A FEW IMAGES FROM THE DATASET
# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize to [0, 1]
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# -- NET DEFINITION
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # feature extraction part
        self.conv1 = nn.Conv2d(3, 6, 5) # 3i, 3o, 5x5k
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # classification part
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -- NETWORK INSTANCE
net = Net()

# -- OPTIMISATION CRITERION
criterion = nn.CrossEntropyLoss()

# -- OPTIMISER
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# -- TRAINING LOOP
if do_training:
    NEPOCHS = 2
    for epoch in range(NEPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            # get data and labels
            inputs, labels = data

            # process input image through network
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            print(outputs, labels)

            # backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # -- SAVE MODEL
    torch.save(net.state_dict(), MODEL_PATH)

# -- OPEN MODEL AND TEST IT
net = Net()
net.load_state_dict(torch.load(MODEL_PATH))

dataiter = iter(testloader)
images, labels = dataiter.next(); print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1); print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# -- GLOBAL EVALUATION
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# -- SPECIFIC EVALUATION
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# -- CUDA GPU PROCESSING
# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)