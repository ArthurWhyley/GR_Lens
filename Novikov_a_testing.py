import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7
M = 1e2

#Scale factor definition

def B(r,t):
    bracket = 2/3 - t / (2 * M**1.5 * (1 + r**2)**1.5)
    if bracket < 0:
        B = 0
    else:
        B = bracket**(1/3)
    return(B)

def a(r,t):
    a = 1.5**(2/3) * B(r,t)**2
    #a = H0 * t**0.5
    return(a)

#Plot a vs r at different t

tnum = 5
tfac = 1
rnum = 1000

rlist = np.arange(rnum) * 0.005 - 2.5
alist = np.zeros([tnum,rnum])
for i in range(tnum):
    for j in range(rnum):
        alist[i,j] = a(rlist[j], i * tfac)
print(alist)

legend = []
for i in range(tnum):
    legend.append("t = " + str(i * tfac))

for i in range(tnum):
    plt.plot(rlist, alist[i])
plt.xlabel("r")
plt.ylabel("a")
plt.legend(legend)
plt.show()

#R = ar vs r

Rlist = alist * rlist
for i in range(tnum):
    plt.plot(rlist, Rlist[i])
plt.xlabel("r")
plt.ylabel("R = ar")
plt.legend(legend)
plt.show()

#a vs R
for i in range(tnum):
    plt.plot(Rlist[i], alist[i])
plt.xlabel("R")
plt.ylabel("a")
plt.legend(legend)
plt.show()