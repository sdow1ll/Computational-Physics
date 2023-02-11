#Exam 1: Problem 4
import numpy as np
import matplotlib.pyplot as plt

def SIR(SIRvar, t, param):
    beta = param[0]
    gamma = param[1]
    
    S = SIRvar[0]
    I = SIRvar[1]
    R = SIRvar[2]

    
    dSdt = -beta* S * I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I
    
    return np.array([dSdt, dIdt, dRdt])

def rk4(x,t,tau,derivsRK,param):
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
    F1 = derivsRK(x,t,param)  
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half,param)  
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half,param)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full,param)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout

tau = 0.01
tplot = np.arange(0, 30, tau)

Splot = np.zeros(len(tplot))
Iplot = np.zeros(len(tplot))
Rplot = np.zeros(len(tplot))
betas = []
Iplotfrac = []

for j in range(2,11):
    SIRvar = np.array([0.9999, 0.001, 0])
    gamma = 1
    param = np.array([j, gamma])
    betas.append(param[0])
    for i in range(len(tplot)):
        SIRvar = rk4(SIRvar, t, tau, SIR, param)
        
        Splot[i] = SIRvar[0]
        Iplot[i] = SIRvar[1]
        Rplot[i] = SIRvar[2]
    
    print('')    
    print('Fraction of infected popul ever for beta =', j ,':', Iplot[-1]) #Iplot[-1] is when t = 30
    Iplotfrac.append(Iplot[-1])                                            #because its the last index
    Iplotmax = np.argmax(Iplot)
    print('Infected peak at:', Iplotmax, 'iterations for beta =', j)
    #plt.figure(j)   
    #plt.plot(tplot, Splot, 'r', label='Susceptible')
    #plt.plot(tplot, Iplot, 'b', label='Infected')
    #plt.plot(tplot, Rplot, 'g', label='Recovered')
    #plt.legend()


plt.figure(1)
plt.plot(betas, Iplotfrac, 'b*')
plt.xlabel('beta values')
plt.ylabel('fraction of population ever infected for each beta')
plt.title('Fraction of Infected Ever vs. Beta')

Iplotpeaktimes = [tplot[674], tplot[382], tplot[270], tplot[210], tplot[173], 
                      tplot[147], tplot[128], tplot[114], tplot[102]]
plt.figure(2)
plt.plot(betas, Iplotpeaktimes, 'r*')   
plt.xlabel('beta values')
plt.ylabel('time when the Infection plot peaks (sec)')
plt.title('Time vs. Beta')