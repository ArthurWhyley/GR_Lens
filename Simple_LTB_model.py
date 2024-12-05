import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
k = 0
c = 1 #km/s
OM = 0.3
OLambda = 0.7

#Scale factor definition

def a(t):
    a = np.e**(H0 * t)
    #a = H0 * t**0.5
    return(a)

def dadt(t):
    dadt = H0 * np.e**(H0 * t)
    #dadt = H0 * 0.5 * t**(-0.5)
    return(dadt)

#Define 4 first order differential equations to solve simultaneously

def t_dot(T):
    #print(T)
    return(T)

def r_dot(R):
    #print(R)
    return(R)

def T_dot(t,r,R):
    if abs(R) < 1e-80:
        T_dot = 0
    else:
        #scales = a(t) * dadt(t)
        lnscales = 2 * t * np.log(H0) #dark energy dominated
        if k == 0:
            kterm = 1
        else:
            kterm = 1 - k * r**2
        R = abs(R)
        lnR2 = 2 * np.log(R)
        lnproduct = lnscales + lnR2
        #print(1, lnproduct, t, R)
        product = np.e**lnproduct
        T_dot = -1 / kterm * product
        #print(T_dot)
    return(T_dot)

def R_dot(t,r,T,R):
    if abs(R) < 1e-80:
        R_dot = 0
    else:
        scales = H0 #for dark energy dominated
        if k == 0:
            kterm = 0
        else:
            kterm = k * r / (1 - k * r**2)
        T = abs(T)
        R = abs(R)
        if (T >= 0 and R >= 0) or (T < 0 and R < 0):
            sign = 1
        else:
            sign = -1
        lnproduct = np.log(T) + np.log(R)
        #print(2, lnproduct, T, R)
        product = np.e**lnproduct
        part1 = -2 * scales * product * sign
        if k == 0:
            part2 = 0
        else:
            part2 = -1 * kterm * R**2
        R_dot = part1 + part2
    return(R_dot)

#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,T,R = y
    y_dot = [t_dot(T), r_dot(R), T_dot(t,r,R), R_dot(t,r,T,R)]
    return(y_dot)

#Define some arbitrary initial conditions

y0 = [0, 1, -1, 1]

#Now do the integration

sigma = np.arange(100) * 0.01
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="LSODA", min_step=0, t_eval=sigma)

print(np.shape(sol.y))
print(np.shape(sol.t))
print(sol.status)

plt.plot(sol.y[0], sol.y[1])
plt.xlabel("t")
plt.ylabel("r")
plt.show()

plt.plot(sol.t, sol.y[0])
plt.xlabel("$\sigma$")
plt.ylabel("t")
plt.show()

plt.plot(sol.t, sol.y[1])
plt.xlabel("$\sigma$")
plt.ylabel("r")
#plt.xscale("log")
#plt.yscale("log")
plt.show()

alist = a(sol.y[0])
print(alist)

#plt.plot(sol.y[0], sol.y[1] * alist)
#plt.xlabel("t")
#plt.ylabel("Angular Diameter Distance")
#plt.show()

#Now plot r and t as functions of a for comparison

astep = np.max(alist) / 1000000

amin = np.min(alist)

def rofa(a0,amax):
    a = np.arange(start=a0, stop=amax, step=astep)
    integrand = a**(-2) * (OM * a**(-3) + OLambda)**(-0.5)
    r = np.sum(integrand) / H0 * astep
    return(r)

rs = np.arange(len(alist), dtype=float)
for i in range(len(alist)):
    rs[i] = rofa(amin, alist[i])

plt.plot(alist, rs)
plt.xlabel("a")
plt.ylabel("r")
#plt.xlim([0,10])
plt.show()

#Same again but for t

def tofa(a0,amax):
    a = np.arange(start=a0, stop=amax, step=astep)
    #print(len(a))
    integrand = a**(-1) * (OM * a**(-3) + OLambda)**(-0.5)
    t = np.sum(integrand) / H0 * astep
    return(t)

ts = np.arange(len(alist), dtype=float)
for i in range(len(alist)):
    ts[i] = tofa(amin, alist[i])

plt.plot(alist, ts)
plt.xlabel("a")
plt.ylabel("t")
plt.show()

#Plot a versus sigma

plt.plot(sigma, alist)
plt.xlabel("$\sigma$")
plt.ylabel("a")
plt.show()

#Plot comparison between r vs t from a and from the direct numerical solution

plt.plot(ts, rs)
plt.plot(sol.y[0], sol.y[1])
plt.xlabel("t")
plt.ylabel("r")
plt.legend(["f(a)", "f($\sigma$)"])
#plt.xlim([0,0.01])
#plt.ylim([0,0.01])
plt.show()
