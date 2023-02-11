import numpy as np
import matplotlib.pyplot as plt

def lax(N, tau, freq):
 
    L = 1.      # System size
    h = L/N     # Grid spacing
    c = 1.      # Wave speed
    coeff = -c*tau/(2.*h)    # Coefficient used by all schemes
    coefflw = 2*coeff**2     # Coefficient used by L-W scheme
    nStep = 500
    x = np.arange(N)*h - L/2. 
    
    a = np.zeros(N)
    ip = np.arange(2,50)  
    im = np.arange(0,48)  
    
    iplot = 1           # Plot counter
    nplots = 50     # Desired number of plots
    aplot = np.empty((N,nplots))
    tplot = np.empty(nplots)
    aplot[:,0] = np.copy(a)     # Record the initial state
    tplot[0] = 0                # Record the initial time (t=0)
    plotStep = nStep/nplots +1  # Number of steps between plots
    
    for iStep in range(nStep):
        a[0] = np.sin(freq*tau*iStep)
        a[-1] = 0
        a[1:49] = .5*( a[ip] + a[im] ) + coeff*( a[ip] - a[im] )   
        if (iStep+1) % plotStep < 1 :        # Every plot_iter steps record 
            aplot[:,iplot] = np.copy(a)      # Record a(i) for ploting
            tplot[iplot] = tau*(iStep+1)
            iplot += 1

            
    plt.plot(x,aplot[:,0],'-',x,a,'--')
    plt.title('Lax Method')
    plt.legend(['Initial  ','Final'])
    plt.xlim([-0.5, 0.5])
    plt.xlabel('x')  
    plt.ylabel('a(x,t)')
    plt.show()
    
def laxwendroff(N, tau, freq):
 
    L = 1.      # System size
    h = L/N     # Grid spacing
    c = 1.      # Wave speed
    coeff = -c*tau/(2.*h)    # Coefficient used by all schemes
    coefflw = 2*coeff**2     # Coefficient used by L-W scheme
    nStep = 500
    x = np.arange(N)*h - L/2. 
    
    a = np.zeros(N)
    ip = np.arange(2,50)
    im = np.arange(0,48)
    
    iplot = 1           # Plot counter
    nplots = 50     # Desired number of plots
    aplot = np.empty((N,nplots))
    tplot = np.empty(nplots)
    aplot[:,0] = np.copy(a)     # Record the initial state
    tplot[0] = 0                # Record the initial time (t=0)
    plotStep = nStep/nplots +1  # Number of steps between plots
    
    for iStep in range(nStep):
        a[0] = np.sin(freq*tau*iStep)
        a[-1] = 0
        a[1:49] = ( a[1:49] + coeff*( a[ip] - a[im] ) + 
                coefflw*( a[ip] + a[im] -2*a[1:49] ) ) 
        if (iStep+1) % plotStep < 1 :        # Every plot_iter steps record 
            aplot[:,iplot] = np.copy(a)      # Record a(i) for ploting
            tplot[iplot] = tau*(iStep+1)
            iplot += 1
            
    plt.plot(x,aplot[:,0],'-',x,a,'--')
    plt.legend(['Initial  ','Final'])
    plt.xlabel('x')  
    plt.xlim([-0.5,0.5])
    plt.ylabel('a(x,t)')
    plt.title('Lax Wendroff Method')
    plt.show()
    
N = 50
tau = np.array([0.015, 0.02, 0.03])
freq = np.array([np.pi*10, np.pi*20])

lax(N, tau[0], freq[0])
laxwendroff(N, tau[0], freq[0])


lax(N, tau[1], freq[0])
laxwendroff(N, tau[1], freq[0])
 
lax(N, tau[2], freq[0])
laxwendroff(N, tau[2], freq[0])

lax(N, tau[0], freq[1])
laxwendroff(N, tau[0], freq[1])
