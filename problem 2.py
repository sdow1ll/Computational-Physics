#Exam 1 Problem 2:

import numpy as np
import matplotlib.pyplot as plt

def rka(x,t,tau,err,derivsRK,param):
    """Adaptive Runge-Kutta routine
       Inputs
        x          Current value of the dependent variable
        t          Independent variable (usually time)
        tau        Step size (usually time step)
        err        Desired fractional local truncation error
        derivsRK   Right hand side of the ODE; derivsRK is the
                   name of the function which returns dx/dt
                   Calling format derivsRK (x,t,param).
        param      Extra parameters passed to derivsRK
       Outputs
        xSmall     New value of the dependent variable
        t          New value of the independent variable
        tau        Suggested step size for next call to rka
    """
    
    #* Set initial variables
    tSave, xSave = t, x        # Save initial values
    safe1, safe2 = 0.9, 4.0    # Safety factors
    eps = 1.e-15

    #* Loop over maximum number of attempts to satisfy error bound
    xTemp = np.empty(len(x))
    xSmall = np.empty(len(x)); xBig = np.empty(len(x))
    maxTry = 100
    for iTry in range(maxTry):

        #* Take the two small time steps
        half_tau = 0.5 * tau
        xTemp = rk4(xSave,tSave,half_tau,derivsRK,param)
        t = tSave + half_tau
        xSmall = rk4(xTemp,t,half_tau,derivsRK,param)
  
        #* Take the single big time step
        t = tSave + tau
        xBig = rk4(xSave,tSave,tau,derivsRK,param)
  
        #* Compute the estimated truncation error
        scale = err * (abs(xSmall) + abs(xBig))/2.
        xDiff = xSmall - xBig
        errorRatio = np.max( np.absolute(xDiff) / (scale + eps) )
  
        #* Estimate new tau value (including safety factors)
        tau_old = tau
        tau = safe1*tau_old*errorRatio**(-0.20)
        tau = max(tau, tau_old/safe2)
        tau = min(tau, safe2*tau_old)
  
        #* If error is acceptable, return computed values
        if errorRatio < 1:
            return np.array([xSmall, t, tau]) #I took out the print statement because the output
        #was too long.

    #* Issue error message if error bound never satisfied
    print('ERROR: Adaptive Runge-Kutta routine failed')
    return np.array([xSmall, t, tau])
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

def stareqn(states, t, param):
    " Inputs"
    "     s1      State vector for star 1 [r1(1) r1(2) v1(1) v1(2)]"
    "     s2      State vector for star 2 [r2(1) r2(2) v2(1) v2(2)]"
    "     s3      State vector for star 3 [r3(1) r3(2) v3(1) v3(2)]"
    "     G     Parameter G gravitational constant"
    "Output"
    "     deriv  Derivatives [dr1(1)/dt dr1(2)/dt dv1(1)/dt dv1(2)/dt]"
    G = param[0]
    m1 = param[1]
    m2 = param[2]
    m3 = param[3]
    
    r1x = states[0]
    r1y = states[1]
    v1x = states[2]
    v1y = states[3]
    r1 =  np.array([r1x, r1y])
    v1 =  np.array([v1x, v1y])
    
    r2x = states[4] 
    r2y = states[5]
    v2x = states[6]
    v2y = states[7]
    r2 =  np.array([r2x, r2y])
    v2 =  np.array([v2x, v2y])
    
    r3x = states[8]
    r3y = states[9]
    v3x = states[10]
    v3y = states[11]
    r3 =  np.array([r3x, r3y])
    v3 =  np.array([v3x, v3y])
    
    star1accel = (G*m2*(r2-r1)/np.linalg.norm((r2-r1))**3)  +  (G*m3*(r3-r1)/np.linalg.norm((r3-r1))**3)
    star2accel = (G*m1*(r1-r2)/np.linalg.norm((r1-r2))**3)  +  (G*m3*(r3-r2)/np.linalg.norm((r3-r2))**3)
    star3accel = (G*m1*(r1-r3)/np.linalg.norm((r1-r3))**3)  +  (G*m2*(r2-r3)/np.linalg.norm((r2-r3))**3)
    
    star1vel = G*v1
    star2vel = G*v2
    star3vel = G*v3
    #deriv1 = np.array([v1[0], v1[1], star1accel[0], star1accel[1]])
    #deriv2 = np.array([v2[0], v2[1], star2accel[0], star2accel[1]])
    #deriv3 = np.array([v3[0], v3[1], star3accel[0], star3accel[1]]) #adjust staraccels to each state
    
    derivArray = np.array([star1vel[0], star1vel[1], star1accel[0], star1accel[1], 
                           star2vel[0], star2vel[1], star2accel[0], star2accel[1],
                           star3vel[0], star3vel[1], star3accel[0], star3accel[1]])
    return derivArray

m1 = 150 
m2 = 200
m3 = 250
G = 1
param = np.array([G, m1, m2, m3])

tau = 0.01
time = 0
tplot = np.arange(0, 2, tau)

xplot1 = np.zeros(len(tplot))
yplot1 = np.zeros(len(tplot))
xplot2 = np.zeros(len(tplot))
yplot2 = np.zeros(len(tplot))
xplot3 = np.zeros(len(tplot))
yplot3 = np.zeros(len(tplot))

r1 = np.array([3, 1])
r2 = np.array([-1, 2])
r3 = np.array([-1, 1])
s1vel = np.array([0, 0])
s2vel = np.array([0, 0])
s3vel = np.array([0, 0])

states = np.array([r1[0], r1[1], s1vel[0], s1vel[1],
                   r2[0], r2[1], s2vel[0], s2vel[1],
                   r3[0], r3[1], s3vel[0], s3vel[1]])
param = np.array([G, m1, m2, m3])
adaptErr = 1e-3
    
for i in range(len(tplot)):
        xplot1[i] = states[0] 
        yplot1[i] = states[1]
        
        xplot2[i] = states[4]
        yplot2[i] = states[5]
        
        xplot3[i] = states[8] 
        yplot3[i] = states[9]
        
        states = rk4(states,time,tau,stareqn,param)
        
        #r1 = np.array([states[0], states[1]])   # 4th order Runge-Kutta
        #v1 = np.array([states[2], states[3]])
        #r2 = np.array([states[4], states[5]])
        #v2 = np.array([states[6], states[7]])
        #r3 = np.array([states[8], states[9]])
        #v3 = np.array([states[10], states[11]])
        
        #[states, time, tau] = rka(states,time,tau,adaptErr,stareqn,param) #rka func
plt.plot(xplot1, yplot1)
plt.plot(xplot2, yplot2)
plt.plot(xplot3, yplot3)
plt.title('Star Trajectories')
plt.xlabel('x')
plt.ylabel('y')