import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7
M = 1e6

#k and J are set by Novikov initial conditions

PHi = -0.0011111111111111113
ri = -1000
Ri = -100

J = PHi * ri**2
k = (Ri**2 + (1 - 2 * M / ri) * J**2 / ri**2)**0.5

#Define 4 first order differential equations to solve simultaneously

def r_dot(R):
    return(R)

def t_dot(r):
    #r = abs(r)
    t_dot = k / (1 - 2 * M / r)
    return(t_dot)

def R_dot(r,R):
    #r = abs(r)
    term1 = -1 * (1 - 2 * M / r) * M * r**-2 * t_dot(r)**2
    term2 = (1 - 2 * M / r)**-1 * M * r**-2 * R**2
    term3 = (1 - 2 * M / r) * r * phi_dot(r)**2
    print(term1, term2, term3)
    R_dot = term1 + term2 + term3 
    return(R_dot)

def phi_dot(r):
    phi_dot = J / r**2
    return(phi_dot)

#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,phi,R = y
    y_dot = [t_dot(r), r_dot(r), phi_dot(r), R_dot(r,R)]
    return(y_dot)

#Initial conditions

y0 = [0, ri, -0.01, Ri]

#Integration

sigma = np.arange(100000) * 0.00001 
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="LSODA", min_step=0, max_step=100, t_eval=sigma)

#Plot xy

x = sol.y[1] * np.cos(sol.y[2])
y = sol.y[1] * np.sin(sol.y[2])

plt.plot(x,y)
plt.scatter(0,0, marker="x")
plt.xlabel("x")
plt.ylabel("y")
#plt.xlim([-1000,1000])
#plt.ylim([-100,100])
plt.show()

print(R_dot(ri,Ri),Ri,ri)

#Plot t, r and phi vs sigma

plt.plot(sol.t,sol.y[0])
#plt.plot(sol.t,sol.y[1])
plt.plot(sol.t,sol.y[2])
plt.xlabel("$\sigma$")
plt.ylabel("value")
plt.legend(["t","r","$\phi$"])
plt.show()

plt.plot(sol.t,sol.y[3])
plt.xlabel("$\sigma$")
plt.ylabel("r_dot")
plt.show()

plt.plot(sol.y[1],sol.y[3])
plt.xlabel("$r$")
plt.ylabel("r_dot")
plt.show()
