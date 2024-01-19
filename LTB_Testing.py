import numpy as np
from matplotlib import pyplot as plt

#All equations based on those found in Enqvist 2007

#Define a distance-dependant H function
def H0(r):
    H0 = 130 * np.cos(r)
    return(H0)

#Define a distance-dependant omega M
def OM(r):
    OM = 0.3 * 1.01**r
    return(OM)

#Define a distance-dependant omega lambda
def OL(r):
    OL = 0.69 * 1.01**r
    return(OL)

#Scale factor
A0 = 1
def A(r,t):
    A = 1.01**t + 0.001 * np.sin(r)
    return(A)

#Define a function for H squared
def H_sq(r,t):
    OC = 1 - OL(r) - OM(r)
    H_sq = H0(r)**2 * (OM(r) * (A0 / A(r,t))**3 + OL(r) + OC * (A0 / A(r,t))**2)
    return(H_sq)

r = (np.arange(10) + 1) * 10
t = np.arange(100)
for j in range(len(r)):
    H2 = np.zeros(len(t))
    for i in range(len(t)):
        H2[i] = H_sq(r[j], t[i])
    H = H2**0.5
    plt.plot(t,H)

plt.xlabel("t")
plt.ylabel("H")
plt.legend(r)
plt.show()