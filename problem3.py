import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def p3(tau, kappa):
    N = 50
    L = 1
    h = 0.02
    c = 1
    nStep = 50
    
#* Set initial and boundary conditions.

    x = np.arange(N)*h - L/2.   # Coordinates of grid points
# Initial condition is dirac delta
    a = np.zeros(N)
    for i in range(N) :
        a[int(N/2)] = 1./h
        
    
# Use periodic boundary conditions
    ip = np.arange(N) + 1  
    ip[N-1] = 0          # ip = i+1 with periodic b.c.
    im = np.arange(N) - 1  
    im[0] = N-1          # im = i-1 with periodic b.c.

#* Initialize plotting variables.
    iplot = 1           # Plot counter
    nplots = nStep     # Desired number of plots
    aplot = np.empty((N,nplots))
    tplot = np.empty(nplots)
    aplot[:,0] = np.copy(a)     # Record the initial state
    tplot[0] = 0                # Record the initial time (t=0)
    plotStep = nStep/nplots +1  # Number of steps between plots

#* Loop over desired number of steps.

    for iStep in range(nStep):  ## MAIN LOOP ##
        coeff1 = -c/tau*2*h
        coeff2 = kappa/tau*h**2
        a[:] = coeff1*(a[ip] - a[im]) + coeff2*(a[ip] + a[im] - 2*a[:]) + a[:]
        
        #     #* Periodically record a(t) for plotting.
        if (iStep+1) % plotStep < 1 :        # Every plot_iter steps record 
            aplot[:,iplot] = np.copy(a)      # Record a(i) for ploting
            tplot[iplot] = tau*(iStep+1)
            iplot += 1
            
            
            #* Plot the initial and final states.
    plt.figure()
    plt.grid()
    plt.plot(x,a,'--')
    plt.plot(x,aplot[:,0])
    plt.legend(['Initial  ','Final'])
    plt.xlabel('x')  
    plt.ylabel('a(x,t)')
    plt.title("initial & final states, tau = {}".format(tau))
    plt.show()
  

#b)
#different taus to try for kappa=1
p3(1.1, 1)
p3(0.01, 1)
p3(0.001, 1)
p3(0.0001, 1)


#c
def p3time(t):
    N = 50
    L = 1
    h = 0.02
    c = 1
    nStep = 101
    kappa= 0.01
    tau = 0.01
#* Set initial and boundary conditions.

    x = np.arange(N)*h - L/2.   # Coordinates of grid points
# Initial condition is dirac delta
    a = np.zeros(N)
    
    a[int(N/2)] = 1./h
           
# Use periodic boundary conditions
    ip = np.arange(N) + 1  
    ip[N-1] = 0          # ip = i+1 with periodic b.c.
    im = np.arange(N) - 1  
    im[0] = N-1          # im = i-1 with periodic b.c.

#* Initialize plotting variables.
    iplot = 1           # Plot counter
    nplots = nStep     # Desired number of plots
    aplot = np.empty((N,nplots))
    tplot = np.empty(nplots)
    aplot[:,0] = np.copy(a)     # Record the initial state
    tplot[0] = 0                # Record the initial time (t=0)
    plotStep = nStep/nplots +1  # Number of steps between plots

#* Loop over desired number of steps.

    for iStep in range(nStep):  ## MAIN LOOP ##
        coeff1 = -c/2*h*tau
        coeff2 = kappa/tau*h**2
        a[:] = coeff1*(a[ip] - a[im]) + coeff2*(a[ip] + a[im] - 2*a[:]) + a[:]
        
        #     #* Periodically record a(t) for plotting.
        if (iStep+1) % plotStep < 1 :        # Every plot_iter steps record 
            aplot[:,iplot] = np.copy(a)      # Record a(i) for ploting
            tplot[iplot] = tau*(iStep+1)
            iplot += 1
    
            
    plt.plot(x, aplot[:,t])
    plt.xlabel('x')  
    plt.ylabel('a(x,t)')
    plt.title("state at t = 0.{} sec".format(t))

    
plt.figure(10)
p3time(0)
plt.figure(11)
p3time(25)
plt.figure(12)
p3time(50)
plt.figure(13)
p3time(75)
plt.figure(14)
p3time(100) #meant to be 1 second instead of 0.1
