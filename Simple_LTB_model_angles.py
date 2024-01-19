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

#Define 8 first order differential equations to solve simultaneously

def t_dot(T):
    #print(T)
    return(T)

def r_dot(R):
    #print(R)
    return(R)

def theta_dot(TH):
    return(TH)

def phi_dot(PH):
    return(PH)

def T_dot(t,r,R,theta,TH,PH):
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
        Mterm = 0
        angles = -1 * a(t) * dadt(t) * r**2 * (TH**2 + np.sin(theta)**2 * PH**2)
        T_dot = -1 / (kterm - Mterm) * product
        T_dot = T_dot + angles
        #print(T_dot)
    return(T_dot)

def R_dot(t,r,T,R,theta,TH,PH):
    if abs(R) < 1e-80:
        R_dot = 0
    else:
        scales = H0 #for dark energy dominated
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
        if r == 0:
            part2 = 0
        else:
            part2up = k * r**3 
            part2down = r**2 - k * r**4
            part2 = part2up / part2down
            part2 = part2 * R**2
        part3 = r * (1 - k * r**2) * (TH**2 + np.sin(theta)**2 * PH**2)
        R_dot = part1 + part2 + part3
        #print(part1, part2)
    return(R_dot)

def TH_dot(t,r,T,R,theta,TH,PH):
    scales = H0 #for dark energy dominated
    part1 = -2 * scales * T * TH
    part2 = -2 / r * R * TH
    part3 = np.sin(theta) * np.cos(theta) * PH**2
    TH_dot = part1 + part2 + part3
    return(TH_dot)

def PH_dot(t,r,T,R,theta,TH,PH):
    scales = H0 #for dark energy dominated
    part1 = -2 * scales * T * PH
    part2 = -2 / r * R * PH
    part3 = -2 * np.cos(theta) / np.sin(theta) * TH * PH
    PH_dot = part1 + part2 + part3
    return(PH_dot)
    
#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,T,R,theta,phi,TH,PH = y
    y_dot = [t_dot(T), r_dot(R), T_dot(t,r,R,theta,TH,PH), R_dot(t,r,T,R,theta,TH,PH),
             theta_dot(TH), phi_dot(PH), TH_dot(t,r,T,R,theta,TH,PH), PH_dot(t,r,T,R,theta,TH,PH)]
    return(y_dot)

#Define some arbitrary initial conditions

y0 = [0, 0.5, -1, 1, np.pi/2, 0, 0, 0]

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

plt.plot(sol.y[0], sol.y[4])
plt.xlabel("t")
plt.ylabel("$\Theta$")
plt.show()

plt.plot(sol.y[0], sol.y[5])
plt.xlabel("t")
plt.ylabel("$\phi$")
plt.show()

#Plot y and z

alist = a(sol.y[0])

x = alist * sol.y[1] * np.sin(sol.y[4]) * np.cos(sol.y[5])
y = alist * sol.y[1] * np.sin(sol.y[4]) * np.sin(sol.y[5])
z = alist * sol.y[1] * np.cos(sol.y[4])

xc = sol.y[1] * np.sin(sol.y[4]) * np.cos(sol.y[5]) #comoving xyz
yc = sol.y[1] * np.sin(sol.y[4]) * np.sin(sol.y[5])
zc = sol.y[1] * np.cos(sol.y[4])

x0 = alist[0] * sol.y[1][0] * np.sin(sol.y[4][0]) * np.cos(sol.y[5][0])
y0 = alist[0] * sol.y[1][0] * np.sin(sol.y[4][0]) * np.sin(sol.y[5][0])
z0 = alist[0] * sol.y[1][0] * np.cos(sol.y[4][0])

plt.plot(x,y)
plt.plot(xc,yc)
plt.scatter(x0, y0)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Physical", "Comoving", "$r_0$"])
plt.show()

plt.plot(x,z)
plt.plot(xc,zc)
plt.scatter(x0, z0)
plt.xlabel("x")
plt.ylabel("z")
plt.legend(["Physical", "Comoving", "$r_0$"])
plt.show()

plt.plot(y,z)
plt.plot(yc,zc)
plt.scatter(y0, z0)
plt.xlabel("y")
plt.ylabel("z")
plt.legend(["Physical", "Comoving", "$r_0$"])
plt.show()