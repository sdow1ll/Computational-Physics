import numpy as np
import matplotlib.pyplot as plt

def rk4(x,t,tau,derivsRK,A,B):
    """Runge-Kutta integrator (4th order)
       Input arguments -
        x = current value of dependent variable
        t = independent variable (usually time)
        tau = step size (usually timestep)
        derivsRK = right hand side of the ODE; derivsRK is the
                  name of the function which returns dx/dt
                  Calling format derivsRK (x,t,param).
        param = extra parameters passed to derivsRK
       Output arguments -
        xout = new value of x after a step of size tau
    """
    half_tau = 0.5*tau
    F1 = derivsRK(x,t,A,B)  
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half,A,B)  
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half,A,B)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full,A,B)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout

def BelZha(s,t,A,B):
    
    x,y = s[0], s[1]
    

    dxdt = A + x**2*y - (B+1)*x
    dydt = B*x - x**2*y
   
    deriv = np.array([dxdt, dydt])
    return deriv


#part 1
x0 = np.array([0.1, 1, 1])
y0 = np.array([2, 1, 1])


time = 0
tau = 0.01
steps = 10000

xplot1 = []
yplot1 = []
xplot2 = []
yplot2 = []
xplot3 = []
yplot3 = []
tplot = []


state1 = np.array([x0[0], y0[0]])
state2 = np.array([x0[1], y0[1]])
state3 = np.array([x0[2], y0[2]])

A = np.array([0.5, 1, 2])
B = np.array([1, 3, 5])

for iterations in range(steps):  
       
        state1 = rk4(state1,time,tau,BelZha,A[0],B[0])
        state2 = rk4(state2,time,tau,BelZha,A[1],B[1])
        state3 = rk4(state3,time,tau,BelZha,A[2],B[2])
        
        xplot1.append(state1[0])  
        yplot1.append(state1[1])
        
        xplot2.append(state2[0])  
        yplot2.append(state2[1])
        
        xplot3.append(state3[0])  
        yplot3.append(state3[1])
        
        tplot.append(time)
        time = time + tau  

plt.figure(1)
plt.plot(xplot1, yplot1, label='Trajectory')
plt.plot(A[0], B[0]/A[0], 'rx', label='Steady State')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.figure(2)
plt.plot(xplot2, yplot2, label='Trajectory')
plt.plot(A[1], B[1]/A[1], 'rx', label='Steady State')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.figure(3)
plt.plot(xplot3, yplot3, label='Trajectory')
plt.plot(A[2], B[2]/A[2], 'rx', label='Steady State')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.figure(4)
plt.plot(tplot, xplot1, label='Trajectory')
ArrayA = np.empty(len(tplot))
ArrayA.fill(A[0])
plt.plot(tplot, ArrayA, color='r', ls='dashed', label='Steady State')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()

plt.figure(5)
plt.plot(tplot, xplot2, label='Trajectory')
ArrayA1 = np.empty(len(tplot))
ArrayA1.fill(A[1])
plt.plot(tplot, ArrayA1, color='r', ls='dashed', label='Steady State')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()

plt.figure(6)
plt.plot(tplot, xplot3, label='Trajectory')
ArrayA2 = np.empty(len(tplot))
ArrayA2.fill(A[2])
plt.plot(tplot, ArrayA2, color='r', ls='dashed', label='Steady State')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()