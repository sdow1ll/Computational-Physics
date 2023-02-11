#Exam 1 Problem 1:

import numpy as np
import matplotlib.pyplot as plt

def fnewt(x,param):
    """Function used by the N-variable Newton's method
       Inputs
         x:     State vector [v1 v2]
         param:     Parameters [R1 R2 R3 R4 V+ I0 VT]
       Outputs
         f     Circuit with diode r.h.s. [dv1/dt dv2/dt]
         D     Jacobian matrix, D(i,j) = df(j)/dx(i)
    """
    
    # Evaluate f(i)
    f = np.empty(2) #circuit eqns
    f[0] = (- param[5] * np.exp( (x[0] - x[1]) / param[6] )) + (param[4]/param[0]) - (x[0]/param[1]) #function evaled at intial guess
    f[1] = (param[4]/param[2]) + (param[5] * np.exp( (x[0] - x[1]) / param[6] )) - (x[1]/param[3])

    # Evaluate D(i,j)
    # D matrix is transpose of J
    D = np.empty((2,2))
    D[0,0] = -param[5] * (np.exp( (x[0] - x[1]) / param[6] ) / param[6] ) - ( 1/param[1] )    # df(0)/dx(0)
    D[0,1] = param[5] * (np.exp( (x[0] - x[1]) / param[6] ) / param[6] )  # df(1)/dx(0)
    D[1,0] = param[5] * (np.exp( (x[0] - x[1]) / param[6] ) / param[6] )    # df(0)/dx(1)
    D[1,1] = -param[5] * (np.exp( (x[0] - x[1]) / param[6] ) / param[6] )  - ( 1/param[3] )    # df(1)/dx(1)
    return [f, D]

vplus = 10 #volts
vT = 0.05 #volts
I0 = 3e-9 #amps
R1 = 1e3 #ohms
R2 = 4e3 #ohms
R3 = 3e3 #ohms
R4 = 2e3 #ohms

x0 = np.array(  [ 1, 0 ]  )
x = np.copy(x0)

param = np.array([R1, R2, R3, R4, vplus, I0, vT]) # parameters in the circuit

nStep = 20  # Number of iterations before stopping
xp = np.empty((len(x), nStep))
xp[:,0] = np.copy(x[:])     # Record initial guess for plotting

v1plot = np.empty((nStep))
v2plot = np.empty((nStep))

for iStep in range(nStep):

    #* Evaluate function f and its Jacobian matrix D
    [f, D] = fnewt(x,param)       # fnewt returns value of f and D
 
    #* Find dx by Gaussian elimination; transpose D for column vectors
    dx = np.linalg.solve( np.transpose(D), f)    
    
    #* Update the estimate for the root  
    x = x - dx  
    print(x[1])                  # Newton iteration for new x
    xp[:,iStep] = np.copy(x[:])   # Save current estimate for plotting
    v1plot[iStep] = x[0]
    v2plot[iStep] = x[1]
    iters = np.array([x for x in np.arange(0,nStep)])
    iters.reshape(1, nStep)

print('After', nStep, ' iterations [v1, v2] is', x)

plt.figure(1)
plt.plot(iters, v1plot, '*')
plt.xlabel('Iterations')
plt.ylabel('V1 at each iteration')
plt.title('V1')
plt.figure(2)
plt.plot(iters, v2plot, '*')
plt.xlabel('Iterations')
plt.ylabel('V2 at each iteration')
plt.title('V2')

