import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7
M0 = 1e5
Lambda = 3 * OLambda

#Define M(r)

def M(r):
    M = M0 / r**2
    return(M)

def dMdr(r):
    dMdr = -2 * M0 / r**3
    return(dMdr)

#Define E(r)

def E(r):
    E = -2 * M(r) / r
    return(E)

def dEdr(r):
    dEdr = 2 * M(r) / r**2 - 2 * dMdr(r) / r
    return(dEdr)

#t equations

def t_dot(td):
    return(td)

def td_dot(td,r,phid,R,Rd):
    part1 = 2 * Rd**3 / (R * td * (1 + E(r)))
    part2 = -1 * R * Rd / td * phid**2
    td_dot = part1 + part2
    return(td_dot)

#r equations

def r_dot(rd):
    return(rd)

def rd_dot(r,rd,phid,R,Rd):
    part1 = 4 * Rd / R * rd
    part2 = (2 * Rd / rd / R + dEdr(r) / (2 * (1 + E(r)))) * rd**2
    part3 = R * (1 + E(r)) / Rd * rd * phid**2
    rd_dot = part1 + part2 + part3
    return(rd_dot)

#phi equations

def phi_dot(phid):
    return(phid)

def phid_dot(phid,R,Rd):
    phid_dot = -4 * Rd / R * phid
    return(phid_dot)

#R equations

def R_dot(Rd):
    return(Rd)

def Rd_dot(td,r,R,Rd):
    Rd_dot = -2 * Rd**2 / R
    return(Rd_dot)
    
#def Rd_dot(td,r,R,Rd):
    #Rd_dot = td * (2 * M(r) / R + Lambda / 3 * R**2 + E(r))**0.5
    #return(Rd_dot)

#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,td,r,rd,phi,phid,R,Rd = y
    y_dot = [t_dot(td), td_dot(td,r,phid,R,Rd), r_dot(rd), rd_dot(r,rd,phid,R,Rd),
             phi_dot(phid), phid_dot(phid,R,Rd), R_dot(Rd), Rd_dot(td,r,R,Rd)]
    return(y_dot)

impact = 100
rdi = -1
ri = 1000
phii = -0.1
#PHi = impact / (ri + Ri) - phii
phidi = (impact**2 * rdi**2 / (ri**4 - impact**2 * (1 - 2 * M(ri) / ri) * ri**2))**0.5 * -1
tdi = (16 * M(ri)**2 * ri**2 / (1 - 1 / (1 + ri**2)) * rdi**2 + 4 * M(ri)**2 * (1 + ri**2)**2 * phidi**2)**0.5
tdi = tdi * -1 #Multiply by 1 or -1 depending on what direction in time is needed
y0 = [0, tdi, ri, rdi, phii, phidi, ri, rdi]
#     t------r----------phi----------R-----

#Now do the integration

sigma = np.arange(35000) * 0.01 
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="LSODA", min_step=0, t_eval=None)

print(np.shape(sol.y))
print(np.shape(sol.t))
print(sol.status)

#Plot variables against sigma

plt.plot(sol.t,sol.y[2])
plt.xlabel("$\sigma$")
plt.ylabel("r")
plt.show()

plt.plot(sol.t,sol.y[4])
plt.xlabel("$\sigma$")
plt.ylabel("$\phi$")
plt.show()

plt.plot(sol.t,sol.y[6])
plt.xlabel("$\sigma$")
plt.ylabel("R")
plt.show()

#Plot physical xy

x = sol.y[6] * np.cos(sol.y[4])
y = sol.y[6] * np.sin(sol.y[4])

xin = sol.y[6][0] * np.cos(sol.y[4][0])
yin = sol.y[6][0] * np.sin(sol.y[4][0])

plt.plot(x,y)
plt.scatter(0,0, marker="x")
plt.scatter(xin,yin)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Physical")
plt.show()

#Plot comoving xy

x = sol.y[2] * np.cos(sol.y[4])
y = sol.y[2] * np.sin(sol.y[4])

xin = sol.y[2][0] * np.cos(sol.y[4][0])
yin = sol.y[2][0] * np.sin(sol.y[4][0])

plt.plot(x,y)
plt.scatter(0,0, marker="x")
plt.scatter(xin,yin)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comoving")
plt.show()