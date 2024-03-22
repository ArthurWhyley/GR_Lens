import numpy as np
from matplotlib import pyplot as plt

M = 1e1
J = -100.004936311667047
k = -1.004936311667047

r = np.arange(87.5,1000)
rdot2 = k**2 - (1 - 2 * M / r) * J**2 / r**2
rdot = rdot2**0.5

plt.plot(r,rdot)
plt.xlabel("r")
plt.ylabel("rdot")
plt.show()

print(rdot[0])