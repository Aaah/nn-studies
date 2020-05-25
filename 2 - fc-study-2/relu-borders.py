import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np 

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Grab some test data.
# X, Y, Z = mplot3d.axes3d.get_test_data(0.05)

# print(X)
# print(Y)
# print(Z)

# Plot a basic wireframe.
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# plt.show()

def ReLU(x, a, b):
    """Rectified Linear Unit, with bias and offset"""
    return max(0, a * (x-b))

a1 = 1.0; b1 = 1.0
a2 = 0.6; b2 = 0.25

MAPSIZE = 256
BOUND = 5.0
X = np.linspace(-BOUND, BOUND, MAPSIZE)
Y = np.linspace(-BOUND, BOUND, MAPSIZE)
Z = np.zeros((MAPSIZE, MAPSIZE))

for n in range(MAPSIZE):
    for m in range(MAPSIZE):
        Z[n,m] = ReLU(X[n], a1, b1) + ReLU(Y[m], a2, b2)

X = np.tile(X, [MAPSIZE, 1])

plt.figure(figsize=(8,8))
ax = plt.subplot(1, 1, 1, projection='3d')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.plot_wireframe(X, X.T, Z, color="red")
plt.xlabel("input 1")
plt.ylabel("input 2")
ax.set_zlabel('output')
plt.tight_layout()
plt.show()