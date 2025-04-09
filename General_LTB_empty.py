import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0
OLambda = 0.7
M0 = 0
Lambda = 0

#Define M(r)

def M(r):
    M = 0
    return(M)

def dMdr(r):
    dMdr = 0
    return(dMdr)

def d2Mdr2(r):
    d2Mdr2 = 0
    return(d2Mdr2)

#Define E(r)

def E(r):
    E = 0
    return(E)

def dEdr(r):
    dEdr = 0
    return(dEdr)

def d2Edr2(r):
    d2Edr2 = 0
    return(d2Edr2)

#Derivatives of R (and f functions to simplify)

def Rcommat(r,R):
    Rcommat2 = 2 * M(r) / R - Lambda / 3 * R**2 + E(r)
    Rcommat = Rcommat2**0.5
    return(Rcommat)

def Rcommar(r,R):
    Rcommar = 1
    return(Rcommar)

def Rcommart(r,R):
    Rcommart = 0
    return(Rcommart)

def Rcommarr(r,R):
    Rcommarr = 0
    return(Rcommarr)

#Differential equations to solve

def t_dot(td):
    return(td)

def td_dot(r,rd,phid,R):
    part1 = -1 * Rcommar(r,R) * Rcommart(r,R) / (1 + E(r)) * rd**2
    part2 = -1 * R * Rcommat(r,R) * phid**2
    td_dot = part1 + part2
    return(td_dot)

def r_dot(rd):
    return(rd)

def rd_dot(td,r,rd,phid,R):
    part1 = -2 * Rcommart(r,R) / Rcommar(r,R) * td * rd
    part2 = -1 * (Rcommarr(r,R) / Rcommar(r,R) - dEdr(r) / (2 * (1 + E(r)))) * rd**2
    part3 = R * (1 + E(r)) / Rcommar(r,R) * phid**2
    rd_dot = part1 + part2 + part3
    return(rd_dot)

def phi_dot(phid):
    return(phid)

def phid_dot(td,r,rd,phid,R):
    part1 = -2 * Rcommat(r,R) / R * td * phid
    part2 = -2 * Rcommar(r,R) / R * rd * phid
    phid_dot = part1 + part2
    return(phid_dot)

def R_dot(td,r,rd,R):
    R_dot = Rcommat(r,R) * td + Rcommar(r,R) * rd
    return(R_dot)

#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,td,r,rd,phi,phid,R = y
    y_dot = [t_dot(td), td_dot(r,rd,phid,R), r_dot(rd), rd_dot(td,r,rd,phid,R),
             phi_dot(phid), phid_dot(td,r,rd,phid,R), R_dot(td,r,rd,R)]
    #plt.scatter(R*np.cos(phi), R*np.sin(phi))
    #plt.scatter(r*np.cos(phi), r*np.sin(phi))
    #plt.scatter(R,t)
    #plt.scatter(r,t)
    #plt.xlim([-6,100])
    #plt.ylim([-2,100])
    #plt.show()
    null_condition = td**2 - (Rcommar(r,R)**2 / (1 + E(r)) * rd**2) - R**2 * phid**2
    yd = rd * np.sin(phi) + r * np.cos(phi) * phid
    #print(null_condition, yd)
    return(y_dot)

ri = 100
rdi = -1
Ri = ri
phii = 1

AQuad = 1 - np.tan(phii)**2 * Rcommar(ri,Ri)**2
BQuad = -2 * np.tan(phii)**2 * Rcommar(ri,Ri) * Rcommat(ri,Ri) * rdi
CQuad = -1 * Rcommar(ri,Ri)**2 / (1 + E(ri)) * rdi**2 - np.tan(phii)**2 * Rcommat(ri,Ri)**2 * rdi**2

#Original ydot = 0 method using physical y
#tdi = np.roots([AQuad,BQuad,CQuad])
#tdi = min(tdi)      #min or max for positive / negative t evolution
#phidi = -1 * (Rcommar(ri,Ri) * tdi + Rcommat(ri,Ri) * rdi) * np.sin(phii) / (Ri * np.cos(phii))

#Different ydot = 0 method using comoving y
#phidi = 1
phidi =  -1 * rdi * np.sin(phii) / (ri * np.cos(phii))
tdi = -1 * (Rcommar(ri,Ri)**2 / (1 + E(ri)) * rdi**2 + Ri**2 * rdi**2 / ri**2 * np.tan(phii)**2)**0.5

#tdi = -1 * (Rcommar(ri,Ri)**2 / (1 + E(ri)) * rdi**2 + Ri**2 * phidi**2)**0.5

y0 = [0, tdi, ri, rdi, phii, phidi, Ri]
#     t------r----------phi----------R
print(y0)

ydoti = (Rcommar(ri,Ri) * tdi + Rcommat(ri,Ri) * rdi) * np.sin(phii) + Ri * np.cos(phii) * phidi

#Now do the integration

sigma = np.arange(10000) * 1.5e-2
sol = solve_ivp(overall, [0,np.max(sigma)], y0, method="LSODA", min_step=0, t_eval=None)

print(np.shape(sol.y))
print(np.shape(sol.t))
if sol.status == -1: print("INTEGRATION FAILED") 

#Plot variables against sigma

plt.plot(sol.t,sol.y[0])
plt.xlabel("$\sigma$")
plt.ylabel("t")
plt.show()

plt.plot(sol.t,sol.y[2])
plt.xlabel("$\sigma$")
plt.ylabel("r")
plt.show()

plt.plot(sol.t,sol.y[3])
plt.xlabel("$\sigma$")
plt.ylabel("r dot")
plt.show()

plt.plot(sol.t,sol.y[4])
plt.xlabel("$\sigma$")
plt.ylabel("$\phi$")
plt.show()

plt.plot(sol.t,sol.y[5])
plt.xlabel("$\sigma$")
plt.ylabel("$\phi$ dot")
plt.show()

plt.plot(sol.t,sol.y[6])
plt.xlabel("$\sigma$")
plt.ylabel("R")
plt.show()

#Check y dot

#ydot = ((Rcommar(sol.y[2],sol.y[6]) * sol.y[1] + Rcommat(sol.y[2],sol.y[6]) * sol.y[3]) * np.sin(sol.y[4]) +
        #sol.y[6] * np.cos(sol.y[4]) * sol.y[5])
ydot = sol.y[3] * np.sin(sol.y[4]) + sol.y[2] * np.cos(sol.y[4]) * sol.y[5]
plt.plot(sol.t,ydot)
plt.xlabel("$\sigma$")
plt.ylabel("y dot")
#plt.xlim([0,1e-6])
#plt.ylim([-1000000,0])
plt.show()

#Check null condition

#null = (sol.y[1]**2 - Rcommar(sol.y[2],sol.y[6])**2 / (1 + E(sol.y[2])) * sol.y[3]**2 -
#        sol.y[6]**2 * sol.y[5]**2)
#plt.plot(sol.t,null)
#plt.xlabel("$\sigma$")
#plt.ylabel("null condition")
#plt.xlim([0.00011,0.00012])
#plt.yscale("log")
#plt.show()

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
#plt.xlim([54.02,54.04])
#plt.ylim([84.12,84.16])
plt.savefig("Empty_Phys.jpeg", dpi=200)
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
plt.xlim([54.02,54.04])
plt.ylim([84.12,84.16])
plt.show()

print("phii", phii, sol.y[4][0])
print("phidi", phidi, sol.y[5][0])
print("ri", ri, sol.y[2][0])
print("rdi", rdi, sol.y[3][0])
print("ti", 0, sol.y[0][0])
print("tdi", tdi, sol.y[1][0])
print("Ri", Ri, sol.y[6][0])
print("ydi", ydoti, ydot[0])

print(sol.y[1][0]**2)
print(Rcommar(sol.y[2][0],sol.y[6][0])**2 / (1 + E(sol.y[2][0])) * sol.y[3][0]**2)
print(sol.y[6][0]**2 * sol.y[5][0]**2)

r_test = rdi * sol.t + ri
phi_test = phidi * sol.t + phii
x_test = r_test * np.cos(phi_test)
y_test = r_test * np.sin(phi_test)

plt.plot(sol.t,sol.y[2])
plt.plot(sol.t,r_test)
plt.xlabel("$\sigma$")
plt.ylabel("r")
plt.show()

plt.plot(sol.t,sol.y[4])
plt.plot(sol.t,phi_test)
plt.xlabel("$\sigma$")
plt.ylabel("$\phi$")
plt.show()

#plt.plot(x,y)
plt.scatter(0,0, marker="x")
plt.scatter(xin,yin)
plt.plot(x_test,y_test)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comoving")
plt.show()

print("Is phid_dot 0 at sig = 0 (when M = 0)?")
phid_dot_test = -2 * Rcommar(ri,Ri) / Ri * rdi * phidi
print(phid_dot_test)