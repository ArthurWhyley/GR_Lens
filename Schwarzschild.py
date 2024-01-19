import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7

#Set a constant mass and h and k constants

M = 1
h = 4.5   #related to angular momentum
k = 1   #related to energy

#Define 3 first order differential equations to solve simultaneously

def t_dot(r):
    t_dot = k / (1 - (2 * M) / r)
    return(t_dot)

def r_dot(r):
    r_dot = k**2 - h**2 / r**2 * (1 - (2 * M) / r)
    return(r_dot)

def phi_dot(r):
    phi_dot = h / r**2
    return(phi_dot)
    
#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,phi = y
    y_dot = [t_dot(r), r_dot(r), phi_dot(r)]
    return(y_dot)

#Define some arbitrary initial conditions

y0 = [0, 3, 2]

#Now do the integration

sigma = np.arange(100) * 0.01 * 10
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="LSODA", min_step=0, t_eval=sigma)

print(np.shape(sol.y))
print(np.shape(sol.t))
print(sol.status)

plt.plot(sol.y[0], sol.y[1])
plt.xlabel("t")
plt.ylabel("r")
plt.show()

#Plot xy

x = sol.y[1] * np.cos(sol.y[2])
y = sol.y[1] * np.sin(sol.y[2])

x0 = sol.y[1][0] * np.cos(sol.y[2][0])
y0 = sol.y[1][0] * np.sin(sol.y[2][0])

plt.plot(x,y)
plt.scatter(0,0)
plt.scatter(x0,y0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()