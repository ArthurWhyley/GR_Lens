import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7
M = 1e0

im = 10

#Define differential eq for phi

def dphidr(r):
    bracket = im**-2 - r**-2 * (1 - 2 * M /r)
    dphidr = r**-2 * bracket**-0.5
    return(dphidr)

#Plot dphidr for varying r

rlist = np.arange(start=-1000, stop=1000, step=10, dtype=float)
dphidrlist = dphidr(rlist)
plt.plot(rlist,dphidrlist)
plt.xlabel("r")
plt.ylabel("dphidr")
plt.show()