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

#Compute phi(r) for a given impact parameeter

def phi(init,end,impact):
    rstep = 0.001
    if init == 0:
        init = 10
    if end == 0:
        end = -1 * 10
    r = np.arange(start=init,stop=end,step=rstep)
    r = r.astype(float)
    integrand = r**-2 * (impact**-2 - r**-2 * (1 - 2 * M /r))**-0.5
    phi = np.sum(integrand) * rstep
    print(phi, init, end)
    return(phi)

#Now calculate the phis between a set of rs to get an r vs phi plot

im = 10

rlist = np.arange(start=-1000, stop=1000, step=10)
philist = np.zeros(len(rlist))
philist[0] = -0.01

x = np.zeros(len(rlist))
y = np.zeros(len(rlist))
x[0] = rlist[0] * np.cos(philist[0])
y[0] = rlist[0] * np.sin(philist[0])

for i in range(len(rlist) - 1):
    philist[i+1] = philist[i] + phi(rlist[i],rlist[i+1],im)
    x[i+1] = rlist[i+1] * np.cos(philist[i+1])
    y[i+1] = rlist[i+1] * np.sin(philist[i+1])

plt.plot(rlist,philist)
plt.xlabel("r")
plt.ylabel("phi")
plt.show()

print(rlist)
print(philist)
print(x)
print(y)

plt.plot(x,y)
plt.scatter(0,0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
