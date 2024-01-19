import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
k = 0
c = 1 #km/s
A = 1
OM = 0.3
OLambda = 0.7

#Scale factor definition

def a(t):
    a = np.e**(H0 * t)
    #a = H0 * t**0.5
    return(a)

#Define differential equations

def r_dot(t):
    r_dot = A / (a(t))**2
    return(r_dot)

def t_dot(t):
    t_dot = A / a(t) / c
    return(t_dot)

#Container function

def both(sigma,y):
    print(sigma,y)
    t,r = y
    y_dot = [t_dot(t), r_dot(t)]
    return(y_dot)

#Integration

y0 = [0,0]

sigma = np.arange(100) * 1
sol = solve_ivp(both, [0,np.max(sigma)], y0, method="LSODA", min_step=0, t_eval=sigma)

print(sol.t)

plt.plot(sol.y[0], sol.y[1])
plt.xlabel("t")
plt.ylabel("$\chi$")
plt.show()

plt.plot(sol.t, sol.y[0])
plt.xlabel("$\sigma$")
plt.ylabel("t")
plt.show()

plt.plot(sol.t, sol.y[1])
plt.xlabel("$\sigma$")
plt.ylabel("$\chi$")
plt.show()

#t(a) and chi(a) comparisons

alist = a(sol.y[0])

astep = np.max(alist) / 1000000

def rofa(a0,amax):
    a = np.arange(start=a0, stop=amax, step=astep)
    integrand = a**(-2) * (OM * a**(-3) + OLambda)**(-0.5)
    r = np.sum(integrand) / H0 * astep
    return(r)

rs = np.arange(len(alist), dtype=float)
for i in range(len(alist)):
    rs[i] = rofa(1, alist[i])

plt.plot(alist, rs)
plt.xlabel("a")
plt.ylabel("$\chi$")
plt.show()

#Same again but for t

def tofa(a0,amax):
    a = np.arange(start=a0, stop=amax, step=astep)
    integrand = a**(-1) * (OM * a**(-3) + OLambda)**(-0.5)
    t = np.sum(integrand) / H0 * astep
    return(t)

ts = np.arange(len(alist), dtype=float)
for i in range(len(alist)):
    ts[i] = tofa(1, alist[i])

plt.plot(alist, ts)
plt.xlabel("a")
plt.ylabel("t")
plt.show()

#Plot a versus sigma

plt.plot(sigma, alist)
plt.xlabel("$\sigma$")
plt.ylabel("a")
plt.show()