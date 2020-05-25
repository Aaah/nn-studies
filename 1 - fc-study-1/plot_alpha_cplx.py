import numpy as np
import matplotlib.pyplot as plt

def complexity2layers(alpha, k1, k2):
    return alpha * k1 + (1-alpha) * k2 + alpha - alpha ** 2


N = 64.0
I = 128.0
O = 2.0
alpha = np.linspace(0.0, 1.0, 1000)
cplx = complexity2layers(alpha, I/N, O/N)

plt.figure(figsize=(8,4))
plt.plot(alpha, cplx)
plt.xlabel("alpha")
plt.ylabel("number of multiplications")
plt.tight_layout()
plt.show()