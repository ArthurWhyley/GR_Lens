import numpy as np
from matplotlib import pyplot as plt

#Define a mass distribution

M0 = 1   #Mass at r=0
sigma = 0.1
width = 2 * sigma**2
M_centre = 0
rstep = 0.0001

def M(r):
    rlist = np.arange(start=0, stop=r, step=rstep)
    dm = M0 * np.e**(-1 * (rlist - M_centre)**2 / width)
    m = np.sum(dm) * rstep
    #print(r,m)
    return(m)

def dMdr(r):
    dmdr = M0 * np.e**(-1 * (r - M_centre)**2 / width)
    #print(r,dmdr)
    return(dmdr)

#Plot their evolution in r

rs = np.arange(start=-1, stop=1, step=0.01)
Ms = np.zeros(len(rs))
for i in range(len(rs)):
    Ms[i] = M(rs[i])
dMs = dMdr(rs)

plt.plot(rs, Ms)
plt.plot(rs, dMs)
plt.xlabel("r")
plt.ylabel("Mass")
plt.legend(["Cumulative", "At r"])
plt.show()

theory = M0 * sigma * (np.pi / 2)**0.5
print("Total mass calculated by function =", np.max(Ms))
print("Theoretical total =", theory)

#Poster
plt.plot(rs, Ms)
plt.plot(rs, dMs)
plt.xlabel("r")
plt.ylabel("Mass")
plt.legend(["Cumulative", "At r"])
plt.xlim([0,1])
plt.savefig("GaussianProfile.png", dpi=500)
plt.show()