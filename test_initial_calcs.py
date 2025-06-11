import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#Constants

H0 = 1
c = 1 #km/s
OM = 0
OLambda = 0.7
M0 = 0
Lambda = -0.7
rho0 = 1e-2 #unfidorm density of universe (arbitrary)
kappa = 1

#Define rho(R,r)

def rho(r,R):
    rho = rho0 * (r / R)**3
    return(rho)

#Define M(r)

def M(r):
    M = M0 / r**2
    return(M)

def dMdr(r):
    dMdr = -2 * M0 / r**3
    return(dMdr)

def d2Mdr2(r):
    d2Mdr2 = 6 * M0 / r**4
    return(d2Mdr2)

#Define E(r)

def E(r):
    E = -2 * M(r) / r + Lambda / 3 * r**2
    return(E)

def dEdr(r):
    dEdr = 2 * M(r) / r**2 - 2 * dMdr(r) / r + Lambda * 2/3 * r
    return(dEdr)

def d2Edr2(r):
    d2Edr2 = -4 * M(r) / r**3 + 4 * dMdr(r) / r**2 - 2 * d2Mdr2(r) / r + Lambda * 2/3
    return(d2Edr2)

#Derivatives of R (and f functions to simplify)
#Also include derivatives of rho here, as they require the R derivatives

def Rcommatt(r,R):      #Here we're calculating R,tt to then calculate R,t instead of going directly to R,t
    Rcommatt = -1 / 3 * Lambda * R - M(r) / R**2
    return(Rcommatt)

def Rcommat(r,R):
    Rcommat2 =  2 * M(r) / R - Lambda / 3 * R**2 + E(r)
    #if type(Rcommat2) == np.float64:
     #   if Rcommat2 < 0:
      #      Rcommat2 = Rcommat2 * -1
    Rcommat = Rcommat2**0.5 * 1    #to test how things change
    return(Rcommat)

def drhodt(r,R):
    drhodt = -3 * rho0 * r**3 * R**-4 * Rcommat(r,R)
    return(drhodt)

def f(r,R):
    f = (2 * drhodt(r,R) / rho(r,R) * R * Rcommat(r,R) + 3 * Rcommat(r,R)**2
         + E(r) + kappa * rho(r,R) * R**2 + Lambda * R**2)
    return(f)

def Rcommar(r,R):
    Rcommar = -1 * dEdr(r) * R / f(r,R)
    return(Rcommar)

ti = 0
ri = 100
rdi = -1
Ri = ri * 1
phii = 1

#Different ydot = 0 method using comoving y
tdi = -1 * (Rcommar(ri,Ri)**2 / (1 + E(ri)) * rdi**2 + Ri**2 * rdi**2 / ri**2 * np.tan(phii)**2)**0.5
Rdi = Ri * rdi / ri + Rcommat(ri,Ri) * tdi
phidi =  -1 * Rdi * np.sin(phii) / (Ri * np.cos(phii))

y0 = [ti, tdi, ri, rdi, phii, phidi, Ri]
#     t------r----------phi----------R
print(y0)

#Print the step by step components of the calculations

print("R,r", Rcommar(ri,Ri))
print("f", f(ri,Ri))
print("R,t", Rcommat(ri,Ri))
print("R,tt", Rcommatt(ri,Ri))
print("E", E(ri))