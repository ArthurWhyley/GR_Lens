import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7
M = 1e0

#k and J are set by Novikov initial conditions

PHi = -0.0011111111111111113
ri = -1000
Ri = 100

J = PHi * ri**2
k = (Ri**2 + (1 - 2 * M / ri) * J**2 / ri**2)**0.5
#J = -0.1
#k = 1

#Define 3 first order differential equations to solve simultaneously

def t_dot(r):
    r = abs(r)
    t_dot = k / (1 - 2 * M / r)
    return(t_dot)

def r_dot(r):
    r = abs(r)
    r_dot2 = k**2 - (1 - 2 * M / r) * J**2 / r**2
    print(r_dot2, r)
    #r_dot2 = abs(r_dot2)
    r_dot = r_dot2**0.5
    return(r_dot)

def phi_dot(r):
    phi_dot = J / r**2
    return(phi_dot)

#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,phi = y
    y_dot = [t_dot(r), r_dot(r), phi_dot(r)]
    return(y_dot)


#Initial conditions

y0 = [0, ri, -0.01]

#Integration

sigma = np.arange(1000) * 0.1
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="RK45", min_step=0, max_step=100, t_eval=sigma)

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

print(np.min(abs(sol.y[1])))

plt.scatter(x,y)
plt.scatter(0,0, marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-20,20])
plt.ylim([-20,20])
plt.show()

plt.plot(x,y)
plt.scatter(0,0, marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.show()

print(J,k)
print(r_dot(ri))

#Plot rdot equation for varying r 

#M=1e0

rs = np.arange(start=-1999,stop=1000,step=2)
rdot2s = k**2 - (1 - 2 * M / abs(rs)) * J**2 / rs**2

plt.plot(rs, rdot2s)
plt.scatter(J/k,k)
plt.xlabel("r")
plt.ylabel("rdot^2")
plt.xlim(-2000,-1000)
plt.ylim([9000,11000])
plt.show()
