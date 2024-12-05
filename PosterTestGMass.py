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

#Define a mass distribution

M0 = 1e2  #Mass at r=0
sigma_G = 0.11
width = 2 * sigma_G**2
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

#Define 6 first order differential equations to solve simultaneously

def t_dot(T):
    #print(T)
    return(T)

def r_dot(R):
    #print(R)
    return(R)

def phi_dot(PH):
    return(PH)

def T_dot(t,r,R,PH):
    part1 = -1 * a(t) * dadt(t) / (1 - k * r**2 - 2 * M(r) / r) * R**2
    part2 = -1 * a(t) * dadt(t) * r**2 * PH**2
    T_dot = part1 + part2
    return(T_dot)

def R_dot(t,r,T,R,PH):
    Mtemp = M(r)
    part1 = -2 * dadt(t) / a(t) * T * R
    part2 = -1 * (dMdr(r) / r - Mtemp / r**2) / (1 - 2 * Mtemp /r) * R**2 #for k=0
    part3 = r * (1 - k * r**2 - 2 * Mtemp / r) * PH**2
    R_dot = part1 + part2 + part3
    return(R_dot)

def PH_dot(t,r,T,R,PH):
    scales = H0 #for dark energy dominated
    part1 = -2 * scales * T * PH
    part2 = -2 / r * R * PH
    PH_dot = part1 + part2
    return(PH_dot)

#Function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,phi,T,R,PH = y
    y_dot = [t_dot(T), r_dot(R), phi_dot(PH),
             T_dot(t,r,R,PH), R_dot(t,r,T,R,PH), PH_dot(t,r,T,R,PH)]
    return(y_dot)

#Define some arbitrary initial conditions
ri = 10
phii = -0.1
Ri = -1
PHi = ri / (ri + Ri) - phii
y0 = [0, ri, phii, -1, Ri, PHi]

#Now do the integration

sigma = np.arange(60) * 0.01 
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="LSODA", min_step=0, t_eval=None)

print(np.shape(sol.y))
print(np.shape(sol.t))
print(sol.status)

#Plot xy

x = sol.y[1] * np.cos(sol.y[2])
y = sol.y[1] * np.sin(sol.y[2])

xin = sol.y[1][0] * np.cos(sol.y[2][0])
yin = sol.y[1][0] * np.sin(sol.y[2][0])

plt.plot(x,y)
plt.scatter(0,0, marker="x")
plt.scatter(xin,yin)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Plot mass distribution
Ms = np.zeros(len(sol.y[1]))
for i in range(len(sol.y[1])):
    Ms[i] = M(sol.y[1][i])
plt.plot(sol.y[1], Ms)
plt.xlabel("r")
plt.ylabel("M")
plt.show()