import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from cardano_method import CubicEquation as Cubic

#Constants

H0 = 1
c = 1 #km/s
OM = 0.3
OLambda = 0.7
M = 1e-4

#Scale factor definition

def B(r,t):
    bracket = 2/3 - t / (2 * M**1.5 * (1 + r**2)**1.5)
    if bracket < 0:
        B = 0   #stays at 0 because observer has reached the centre
        #B = bracket**(1/3)
    else:
        B = bracket**(1/3)
    return(B)

def a(r,t):
    a = 1.5**(2/3) * B(r,t)**2
    #a = H0 * t**0.5
    return(a)

def dadt(r,t):
    dadt = -1 / (2 * M**1.5 * (1 + r**2)**1.5) * a(r,t)**(-0.5)
    #dadt = H0 * 0.5 * t**(-0.5)
    return(dadt)

def dadr(r,t):
    dadr = 1.5**(2/3) * r * t * (M**1.5 * (r**2 + 1)**2.5 * B(r,t))**-1
    return(dadr)

def d2adr2(r,t):
    d2adr2 = 1.5**(2/3) * (-5 * r**2 * t * (M**1.5 * (r**2 + 1)**3.5 * B(r,t))**-1 
                           + t * (M**1.5 * (r**2 + 1)**2.5 * B(r,t))**-1
                           - r**2 * t**2 * (2 * M**3 * (r**2 + 1)**5 * B(r,t)**4)**-1)
    return(d2adr2)

def d2adrdt(r,t):
    d2adrdt = (1.5**(2/3) * r * (M**1.5 * (1 + r**2)**2.5 * B(r,t))**-1
               + r * t / 2 * (2**(2/3) * 3**(1/3) * M**3 * B(r,t)**4 * (1 + r**2)**4)**-1)
    return(d2adrdt)

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
    r = abs(r)
    bigbracket = (dadr(r,t) * d2adrdt(r,t) * (1 + r**2)**2 + 2 * dadr(r,t) * dadt(r,t) * r * (1 + r**2)
                  + 2 * a(r,t) * d2adrdt(r,t) * r * (1 + r**2) + 4 * a(r,t) * dadt(r,t) * r**2)
    term1 = -4 * M**2 / (1 - (1 + r**2)**-1) * bigbracket * R**2
    term2 = -4 * a(r,t) * dadt(r,t) * M**2 * (1 + r**2)**2 * PH**2
    T_dot = term1 + term2
    return(T_dot)

def R_dot(t,r,T,R,PH):
    r = abs(r)
    term1 = (-2 * (d2adrdt(r,t) * (1 + r**2) + 2 * dadt(r,t) * r)
             / (dadr(r,t) * (1 + r**2) + 2 * a(r,t) * r) * T * R)
    term2frac = ((d2adr2(r,t) * (1 + r**2) + 4 * dadr(r,t) * r + 2 * a(r,t))
                 / (dadr(r,t) * (1 + r**2) + 2 * a(r,t) * r))
    term2 = -1 * (term2frac - (r * (r**2 + 1))**-1) * R**2
    term3 = a(r,t) * r**2 / (dadr(r,t) * (1 + r**2) + 2 * a(r,t) * r) * PH**2
    R_dot = term1 + term2 + term3
    return(R_dot)

def PH_dot(t,r,T,R,PH):
    r = abs(r)
    term1 = -2 * dadt(r,t) / a(r,t) * T * PH
    term2 = -2 * (dadr(r,t) * (1 + r**2) + 2 * a(r,t) * r) / (a(r,t) * (1 + r**2)) * R * PH
    PH_dot = term1 + term2
    return(PH_dot)
    
#Now define a function that contains them all

def overall(sigma,y):
    print(sigma, y)
    t,r,phi,T,R,PH = y
    y_dot = [t_dot(T), r_dot(R), phi_dot(PH),
             T_dot(t,r,R,PH), R_dot(t,r,T,R,PH), PH_dot(t,r,T,R,PH)]
    return(y_dot)

#Define some arbitrary initial conditions
impact = 100
Ri = -1
ri = 1000
phii = -0.1
#PHi = impact / (ri + Ri) - phii
PHi = (impact**2 * Ri**2 / (ri**4 - impact**2 * (1 - 2 * M / ri) * ri**2))**0.5 * -1
Ti = (16 * M**2 *ri**2 / (1 - 1 / (1 + ri**2)) * Ri**2 + 4 * M**2 * (1 + ri**2)**2 * PHi**2)**0.5
Ti = Ti * 1 #Multiply by 1 or -1 depending on what direction in time is needed
y0 = [0, ri, phii, Ti, Ri, PHi] 

#Now do the integration

sigma = np.arange(1000000) * 0.001 
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

#Plot physical x and y

alist = np.zeros(len(sol.y[0]))
amin = 1
amin_t = sol.y[0][0]
amin_x = amin * sol.y[1][0] * np.cos(sol.y[2][0])
amin_y = amin * sol.y[1][0] * np.sin(sol.y[2][0])
for i in range(len(sol.y[0])):
    alist[i] = a(sol.y[1][i], sol.y[0][i])
    if alist[i] < amin:
        amin = alist[i]
        amin_t = sol.y[0][i]
        amin_x = amin * sol.y[1][i] * np.cos(sol.y[2][i])
        amin_y = amin * sol.y[1][i] * np.sin(sol.y[2][i])

xphys = x * alist
yphys = y * alist
plt.plot(xphys, yphys)
plt.scatter(0,0, marker="x")
plt.scatter(amin_x,amin_y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Physical Coords")
plt.legend(["Path", "M", "Min a"])
plt.show()

#Plot a vs t

plt.plot(sol.y[0], alist)
plt.xlabel("t")
plt.ylabel("a")
plt.savefig("Novikov_a.png", dpi=500)
plt.show()

plt.plot(sol.y[1], alist)
plt.xlabel("r")
plt.ylabel("a")
#plt.savefig("Novikov_a.png", dpi=500)
plt.show()

print(amin, amin_t)

#Plot comparison of "comoving" and "physical" coords

plt.plot(x,y)
plt.plot(xphys, yphys, linestyle="--")
plt.scatter(0,0, marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Novikov", "Scaled", "M"])
plt.savefig("Novikov.png", dpi=500)
plt.show()

#Now solve Schwarzschild eqs and compare

J = PHi * ri**2
k = (Ri**2 + (1 - 2 * M / ri) * J**2 / ri**2)**0.5 * -1

#Need to find min r using J and M

rmins = Cubic([1, 0, -1 * J**2, 2 * M * J**2])
for i in range(len(rmins.answers)):
    rmins.answers[i] = rmins.answers[i].real
rmin = np.max(rmins.answers)

changex = 0

def t_dot_S(r):
    #r = abs(r)
    t_dot = k / (1 - 2 * M / r)
    return(t_dot)

def r_dot_S(r,phi):
    #r = abs(r)
    x = r * np.cos(phi)
    r_dot2 = k**2 - (1 - 2 * M / r) * J**2 / r**2
    if r_dot2 < 0:
        print("r_dot^2 < 0!!!!")
        print(r, x)
    if x < changex:
        r_dot = r_dot2**0.5
    else:
        r_dot = r_dot2**0.5 * -1
    return(r_dot)

def phi_dot_S(r,phi):
    phi_dot = J / r**2
    x = r * np.cos(phi)
    if x < changex:
        phi_dot = phi_dot * -1
    return(phi_dot)

#Now define a function that contains them all

def overall_S(sigma,y):
    print(sigma, y)
    t,r,phi = y
    y_dot = [t_dot_S(r), r_dot_S(r,phi), phi_dot_S(r,phi)]
    return(y_dot)


#Initial conditions

y0_S = [0, ri, phii]

#Integration

sigma = np.arange(1000000) * 0.002
sol_S = solve_ivp(overall_S, [0,np.max(sigma)], y0_S, method="LSODA", min_step=0, t_eval=sigma)

#Plot xy comparison

x_S = sol_S.y[1] * np.cos(sol_S.y[2])
y_S = sol_S.y[1] * np.sin(sol_S.y[2])

plt.plot(x,y)
plt.plot(x_S, y_S, linestyle="--")
plt.scatter(0,0, marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Novikov", "Schwarzschild", "M"])
plt.xlim(-1000,1000)
plt.ylim(-500,200)
plt.show()

plt.plot(x,y)
plt.plot(xphys, yphys, linestyle=":")
plt.plot(x_S, y_S, linestyle="--")
plt.scatter(0,0, marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Novikov", "Scaled", "Schwarzschild", "M"])
#plt.xlim(-150,150)
#plt.ylim(-150,150)
plt.show()

plt.plot(sol_S.y[0],sol_S.y[1])
plt.xlabel("t")
plt.ylabel("r")
plt.show()

print(J,k,J/k)
print(PHi)

impact_re = (ri**4 * PHi**2 / (Ri**2 + (1 - 2 * M / ri) * PHi**2))**0.5
print(impact_re)

print(rmins.answers)
print(rmin)

print(y0)