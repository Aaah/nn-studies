import torch
import pickle
import torch.nn as nn
import numpy as np
from fcnn import NN_FC, NN_FC_LEAKY
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt

DISPLAY_OUTPUTS = True
DISPLAY_BORDERS = True

DATASET = './DATA/2d-data-pounds.pkl'
NNEURONS = 25

MODELPATH = './TEMP/model_2_%s_BEST.mdl' % (NNEURONS)

# -- LOAD MODEL
NINPUTS = 2
NCLASSES = 2
net = NN_FC_LEAKY(layers = [NINPUTS, NNEURONS, NNEURONS, NCLASSES]).float()

net.load_state_dict(torch.load(MODELPATH))
net.eval()
print(net.fc)

# -- CREATE TEST DATA
MINIBATCHSIZE = 25
GRIDLEN = 500
BOUND = 5.0
_x = np.linspace(-BOUND, +BOUND, GRIDLEN)
_y = np.linspace(-BOUND, +BOUND, GRIDLEN)

X = torch.torch.zeros([GRIDLEN * GRIDLEN, 2], dtype=torch.float)
for i in np.arange(GRIDLEN):
    for j in np.arange(GRIDLEN):
        X[i * GRIDLEN + j] = torch.tensor([_x[i], _y[j]]).float()

img = np.zeros((GRIDLEN, GRIDLEN, 3))
img0 = np.zeros((GRIDLEN, GRIDLEN))
img1 = np.zeros((GRIDLEN, GRIDLEN))
img2 = np.zeros((GRIDLEN, GRIDLEN))

# -- TEST
with torch.no_grad():
    for i in range(0, int(X.shape[0] / MINIBATCHSIZE)):
        idx = range(i * MINIBATCHSIZE, (i+1) * MINIBATCHSIZE)
        _in = torch.tensor(X[idx]).float()
        _out = net(_in)

        for n, o in enumerate(_out):
            x = int(idx[n] / GRIDLEN)
            y = int(idx[n] % GRIDLEN)

            if DISPLAY_OUTPUTS:
                img1[y, x] = o.data.numpy()[0]
                img2[y, x] = o.data.numpy()[1]

            if DISPLAY_BORDERS:
                if o.data.numpy()[0] > o.data.numpy()[1]:
                    img[y, x] =  [1.0, 0.0, 0.0]
                else:
                    img[y, x] = [0.85, 0.85, 0.85]

# img1 -= img1.mean()
# img1 /= absolute(img1).max()
# img1 = (img1 + 1.0) / 2.0
# img1 *= [1.0, 0.0, 0.0]

# img2 -= img2.mean()
# img2 /= absolute(img2).max()
# img2 = (img2 + 1.0) / 2.0
# img2 *= [0.85, 0.85, 0.85]

# plt.figure(figsize=(8,5))
# plt.subplot(121)
# plt.imshow(img1, interpolation='none', extent=[-10.0,+10.0,-10.0,+10.0])
# plt.subplot(122)
# plt.imshow(img2, interpolation='none', extent=[-10.0,+10.0,-10.0,+10.0])
# plt.tight_layout()
# plt.show()

X = np.tile(_x, [GRIDLEN, 1])

plt.figure(figsize=(14,6))

ax = plt.subplot(1, 2, 1, projection='3d')
plt.title("Classe 0")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.plot_wireframe(X, X.T, img1, color="red", linewidth=0.25, label="class 0")
plt.xlabel("input 1")
plt.ylabel("input 2")
ax.set_zlabel('output')
ax.set_zlim(-5.0, 5.0)

ax = plt.subplot(1, 2, 2, projection='3d')
plt.title("Classe 1")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_zlim(-5.0, 5.0)
ax.plot_wireframe(X, X.T, img2, color="gray", linewidth=0.25, label="class 1")
plt.xlabel("input 1")
plt.ylabel("input 2")
ax.set_zlabel('output')

plt.tight_layout()
plt.show()  

if DISPLAY_BORDERS:
    plt.figure(figsize=(8,5))
    plt.imshow(img, interpolation='none', extent=[-10.0,+10.0,-10.0,+10.0])
    plt.tight_layout()
    plt.show()
            