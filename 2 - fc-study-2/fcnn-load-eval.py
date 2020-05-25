import torch
import pickle
import torch.nn as nn
from numpy import array, ceil, arange, linspace
from fcnn import NN_FC
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

DATASET = './DATA/2d-data-pounds.pkl'
NNEURONS = 20

MODELPATH = './TEMP/model_1_%s_BEST.mdl' % (NNEURONS)

# -- LOAD MODEL
NLAYERS = 1
NINPUTS = 2
NCLASSES = 2
net = NN_FC(layers = [NINPUTS, NNEURONS, NCLASSES]).float()

net.load_state_dict(torch.load(MODELPATH))
net.eval()
print(net.fc)

# -- LOAD DATA
MINIBATCHSIZE = 25
X, Y = pickle.load(open(DATASET,'rb'))

# -- EXTRACT EACH TRANSFORMATION FROM THE NETWORK
seq = nn.Sequential(*list(net.fc.children()))

_X1 = {}; _Y1 = {}; _Z1 = {}
_X2 = {}; _Y2 = {}; _Z2 = {}
for n in arange(len(seq)+1):
    _X1[str(n)] = []
    _Y1[str(n)] = []
    _Z1[str(n)] = []
    _X2[str(n)] = []
    _Y2[str(n)] = []
    _Z2[str(n)] = []

# -- TEST
correct = 0.0; total = 0.0
with torch.no_grad():
    for i in range(1, int(ceil(X.shape[0] / MINIBATCHSIZE))):
        idx = range(i * MINIBATCHSIZE, (i+1) * MINIBATCHSIZE)
        _inputs, _groundtruth = torch.tensor(X[idx]).float(), torch.tensor(Y[idx]).float()
        _outputs = net(_inputs)
        _, predicted = torch.max(_outputs.data, 1)
        total += MINIBATCHSIZE
        correct += (predicted == _groundtruth).sum().item()

        # store data
        for n in range(MINIBATCHSIZE):
            # process only features
            _data = 1. * _inputs[n]
            if _groundtruth.data.numpy()[n] == 0:
                _X1["0"].append(_data.data.numpy()[0])
                _Y1["0"].append(_data.data.numpy()[1])
                for n, process in enumerate(seq):
                    _data = process(_data)
                    _tmp = _data.data.numpy()
                    _X1[str(n+1)].append(_tmp[0])
                    _Y1[str(n+1)].append(_tmp[1])
                    if len(_tmp) > 2:
                        _Z1[str(n+1)].append(_tmp[2])
            else:
                _X2["0"].append(_data.data.numpy()[0])
                _Y2["0"].append(_data.data.numpy()[1])
                for n, process in enumerate(seq):
                    _data = process(_data)
                    _tmp = _data.data.numpy()
                    _X2[str(n+1)].append(_tmp[0])
                    _Y2[str(n+1)].append(_tmp[1])
                    if len(_tmp) > 2:
                        _Z2[str(n+1)].append(_tmp[2])

K = len(seq) + 1
plt.figure(figsize=(4*K + 0.2 * (K-1),5))

for n in arange(K):
    if len(_Z1[str(n)]):
        ax = plt.subplot(1, K, n + 1, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.scatter3D(_X1[str(n)], _Y1[str(n)], _Z1[str(n)], color="gray", s=1, label="class 0");
        ax.scatter3D(_X2[str(n)], _Y2[str(n)], _Z2[str(n)], color="red", s=1, label="class 1"); 
    else:
        plt.subplot(1, K, n + 1)
        plt.plot(_X1[str(n)], _Y1[str(n)], '.', color="gray", markersize=1)
        plt.plot(_X2[str(n)], _Y2[str(n)], '.', color="red", markersize=1)
        plt.grid(color='gray', linestyle='--', linewidth=0.7)

    if n == 0:
        plt.title("ORIGINAL")
    elif n == K-1:
        plt.title("OUTPUT")
    else:
        plt.title("FEATURE (layer %d)" % (n))

plt.tight_layout()
plt.show()

# -- STATISTICS RECAP
test_accuracy = round(100.0 * correct / total, 1)
print("> The loaded model has %0.1f%% accuracy" % test_accuracy)

# # -- MAP TRANSFORMATION
# GRIDLEN = 100
# _x = linspace(-10.0, +10.0, GRIDLEN)
# _y = linspace(-10.0, +10.0, GRIDLEN)
# _X1 = []; _Y1 = []
# _X2 = []; _Y2 = []
# _X3 = []; _Y3 = []

# for i in arange(GRIDLEN):
#     for j in arange(GRIDLEN):

#         _data = torch.tensor([_x[i], _y[j]]).float()
#         for process in seq:
#             _data = process(_data)

#         _X1.append(_x[i])
#         _Y1.append(_y[j])
#         _X2.append(_data.data.numpy()[0])
#         _Y2.append(_data.data.numpy()[1])

#         _output = net(torch.tensor([_x[i], _y[j]]).float())
#         _X3.append(_output.data.numpy()[0])
#         _Y3.append(_output.data.numpy()[1])



# plt.figure(figsize=(13,5))
# plt.subplot(131)
# plt.title("ORIGINAL")
# plt.plot(_X1, _Y1, '.', color="gray", markersize=1)
# plt.grid(color='gray', linestyle='--', linewidth=0.7)
# plt.subplot(132)
# plt.title("FEATURE SPACE")
# plt.plot(_X2, _Y2, '.', color="gray", markersize=1, label="class 1")
# plt.grid(color='gray', linestyle='--', linewidth=0.7)
# plt.subplot(133)
# plt.title("OUTPUT")
# plt.plot(_X3, _Y3, '.', color="gray", markersize=1)
# plt.grid(color='gray', linestyle='--', linewidth=0.7)
# plt.tight_layout()
# plt.show()
