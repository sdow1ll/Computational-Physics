import numpy as np
import matplotlib.pyplot as plt

N = 50
L = 1.      # System size
h = L/N     # Grid spacing
c = 1.      # Wave speed
tau = 0.015
coeff = -c*tau/(h)    # Coefficient used by all schemes
coefflw = 2*coeff**2     # Coefficient used by L-W scheme
nStep = 67

#* Set initial and boundary conditions.
sigma = 0.1                  # Width of the Gaussian pulse
k_wave = np.pi/sigma         # Wave number of the cosine
x = np.arange(N)*h - L/2.    # Coordinates of grid points
# Initial condition is a Gaussian-cosine pulse
a = np.empty(N)
for i in range(N) :
    a[i] = np.cos(k_wave*x[i]) * np.exp(-x[i]**2/(2*sigma**2)) 
# Use periodic boundary conditions
ip = np.arange(N) + 1  
ip[N-1] = 0          # ip = i+1 with periodic b.c.
im = np.arange(N) - 1  
im[0] = N-1          # im = i-1 with periodic b.c.

#* Initialize plotting variables.
iplot = 1           # Plot counter
nplots = 50     # Desired number of plots
aplot = np.empty((N,nplots))
tplot = np.empty(nplots)
aplot[:,0] = np.copy(a)     # Record the initial state
tplot[0] = 0                # Record the initial time (t=0)
plotStep = nStep/nplots +1  # Number of steps between plots

#* Loop over desired number of steps.
for iStep in range(nStep):  ## MAIN LOOP ##
    a[:] = a[:] + coeff*( a[:] - a[im] )  

    #* Periodically record a(t) for plotting.
    if (iStep+1) % plotStep < 1 :        # Every plot_iter steps record 
        aplot[:,iplot] = np.copy(a)      # Record a(i) for ploting
        tplot[iplot] = tau*(iStep+1)
        iplot += 1
        print(iStep, ' out of ', nStep, ' steps completed')


#* Plot the initial and final states.
plt.plot(x,aplot[:,0],'-',x,a,'--')
plt.legend(['Initial  ','Final'])
plt.xlabel('x')  
plt.ylabel('a(x,t)')
plt.show()


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot[0:iplot], x)
ax.plot_surface(Tp, Xp, aplot[:,0:iplot], rstride=1, cstride=1, cmap=cm.gray)
ax.view_init(elev=30., azim=190.)
ax.set_ylabel('Position') 
ax.set_xlabel('Time')
ax.set_zlabel('Amplitude')
plt.show()

