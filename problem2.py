import numpy as np
import matplotlib.pyplot as plt


#* Initialize parameters (time step, grid spacing, etc.).
tau = 1 #i set this to one day as the timestep, but the max tau this will be stable is 1.25
N = 40  #setting tau = 1.25 will cause a lost in accuracy in the plot though
L = 20.   #express the length from the surface to 20 m 
h = 0.5   # Grid size
kappa = 0.1    # Diffusion coefficient
coeff = kappa*tau/h**2
period = 365 #days in a year
A = 6
B = -13


#* Set initial and boundary conditions.
tt = np.zeros(N)                # Initialize temperature to zero at all points
tt.fill(6)                      # Initial cond. is 6 deg everywhere except for surface and bottom


#* Set up loop and plot variables.
xplot = -np.arange(N)*h   # Record the x scale for plots
iplot = 0                        # Counter used to count plots
nstep = 6*365                      # Maximum number of iterations (6 years in day units)
nplots = 6*365                      # Number of snapshots 
plot_step = nstep/nplots         # Number of time steps between plots


#* Loop over the desired number of time steps.
ttplot = np.empty((N,nplots))
tplot = np.empty(nplots)
for istep in range(nstep):  ## MAIN LOOP ##
    #boundary conditions:
    tt[0] = A + B*np.cos(2*np.pi*tau*istep/period)
    tt[-1] = 7
    #* Compute new temperature using FTCS scheme.
    tt[1:(N-1)] = ( tt[1:(N-1)] + 
      coeff*( tt[2:N] + tt[0:(N-2)] - 2*tt[1:(N-1)] ) )
    
    #* Periodically record temperature for plotting.
    if (istep+1) % plot_step < 1 :         # Every plot_step steps
        ttplot[:,iplot] = np.copy(tt)      # record tt(i) for plotting
        tplot[iplot] = (istep+1)*tau       # Record time for plots
        iplot += 1


#* Plot temperature versus x and t as a wire-mesh plot

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1)
ax = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot, xplot)
ax.plot_surface(Xp, Tp, ttplot, rstride=2, cstride=2, cmap=cm.inferno)
ax.set_xlabel('y')
ax.set_ylabel('Time')
ax.set_zlabel('T(y,t)')
ax.set_title('Diffusion on Earth')
plt.show()

#the shallowest depth you can keep the pipes is 6m below the surface
plt.figure(2)

plt.plot(xplot,ttplot[:,1825+90],'-', label= '3rd Month')

plt.plot(xplot,ttplot[:,1825+180],'-', label= '6th Month')

plt.plot(xplot,ttplot[:,1825+270],'-', label= '9th Month')

plt.plot(xplot,ttplot[:,1825+360],'-', label= '12th Month')
plt.legend()
plt.title('surface temperature throughout the 6th year')
plt.xlabel('Depth of the Earth')  
plt.ylabel('T(x,t)')

