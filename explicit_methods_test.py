import numpy as np
from matplotlib import pyplot as plt

#Define a function to do the integration using an explicit Adams-Moulton method

def solver(y, t_range, y0):
    #y = system of equations to solve, see example "eq_system" function below
    #t_range = range of affine parameter to integrate over
    #y0 = initial value of y
    class sol:                              #set up class to contain solutions
        t = t_range                         #t values
        y = np.zeros([len(y0),len(t)])      #y values for each function to solve
    sol.y[:,0] = y0         #set first y values to intiial conditions
    print(sol.t, sol.y)
    h = sol.t[1] - sol.t[0] #calculate step size assuming linear spacing
    print(h)
    for i in range(len(sol.t)-1):
        if i == 0:          #next step has simpler calculations for first few steps
            next_step = sol.y[:,i] + h * y(sol.t[i], sol.y[:,i])
        elif i == 1:
            next_step = sol.y[:,i] + h * (1.5 * y(sol.t[i], sol.y[:,i]) - 0.5 * y(sol.t[i-1], sol.y[:,i-1]))
            print("#")
            print(sol.t[i])
            print(y(sol.t[i], sol.y[:,i]))
            print(y(sol.t[i-1], sol.y[:,i-1]))
        elif i == 2:
            next_step = (sol.y[:,i] + h * (23/12 * y(sol.t[i], sol.y[:,i])
                         - 4/3 * y(sol.t[i-1], sol.y[:,i-1]) + 5/12 * y(sol.t[i-2], sol.y[:,i-2])))
        elif i == 3:
            next_step = (sol.y[:,i] + h * (55/24 * y(sol.t[i], sol.y[:,i]) - 59/24 * y(sol.t[i-1], sol.y[:,i-1])
                         + 37/24 * y(sol.t[i-2], sol.y[:,i-2]) - 3/8 * y(sol.t[i-3], sol.y[:,i-3])))
        else:
            next_step = (sol.y[:,i] + h * (1901/720 * y(sol.t[i], sol.y[:,i])
                         - 2774/720 * y(sol.t[i-1], sol.y[:,i-1]) + 2616/720 * y(sol.t[i-2], sol.y[:,i-2])
                         - 1274/720 * y(sol.t[i-3], sol.y[:,i-3]) + 251/720 * y(sol.t[i-4], sol.y[:,i-4])))
        sol.y[:,i+1] = next_step
        print(sol.y[:,i])
        print(next_step - sol.y[:,i])
    return sol

#Example of how to define a system of equations to input

def test_func1(t):   #first define functions for each differential equation
    adot = 2 * t         
    return adot

def test_func2(b):
    bdot = 3
    return bdot

def eq_system(t,y):
    y1, y2 = y                                  #y has a component for each equation to solve
    ydot = [test_func1(t), test_func2(y2)]     #compute ydot for each ode
    ydot = np.array(ydot)
    return ydot

def x2(x):
    return (x**2)

test = solver(eq_system, np.linspace(1,10,10), [1,3])

plt.plot(test.t,test.y[0])
plt.plot(test.t,test.y[1])
plt.plot(test.t,x2(test.t))
plt.ylim([0,101])
plt.show()

print(test.y)