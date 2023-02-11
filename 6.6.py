#6.6

import numpy as np
import matplotlib.pyplot as plt


#* Initialize parameters (time step, grid spacing, etc.).
tau = 1e-4
N = 61
L = 1.        # The system extends from x=-L/2 to x=L/2
h = L/(N-1)   # Grid size
kappa = 1.    # Diffusion coefficient
coeff = kappa*tau/h**2


#* Set initial/boundary conditions.
tt = np.zeros(N)                  # Initialize temperature to zero at all points
tt[int(3*N/4)] = 1./h             # Initial cond. is offset by a 3/4 factor
x = np.arange(N)*h - L/4.

#* Set up loop and plot variables.
xplot = np.arange(N)*h - L/2.    # Record the x scale for plots
iplot = 0                        # Counter used to count plots
nstep = 300                      # Maximum number of iterations
nplots = 50                      # Number of snapshots (plots) to take
plot_step = nstep/nplots         # Number of time steps between plots

t = 0.03 #final time snapshot of method of images solution

#method of images part at t= 0.03:
def TG(x, n, t, kappa, L):
    delx = -((n*L) + ((-1)**n)*(x[0]))*2
    output = np.empty(len(x))
    for i in range(len(x)):    
        output[i] = exp(-((x[i]-delx)**2) / (4*kappa*t)) / (((2*kappa*t)**(1/2)) * (2*np.pi)**(1/2))
    return output


#* Loop over the desired number of time steps.
ttplot = np.empty((N,nplots))
tplot = np.empty(nplots)

for istep in range(nstep):  ## MAIN LOOP ##
    
    #* Compute new temperature using FTCS scheme.
    tt[1:(N-1)] = ( tt[1:(N-1)] + 
      coeff*( tt[2:N] + tt[0:(N-2)] - 2*tt[1:(N-1)] ) )
    tt[0] = tt[1]
    tt[N-1] = tt[N-2] #boundary conditions neumann
    #* Periodically record temperature for plotting.
    if (istep+1) % plot_step < 1 :         # Every plot_step steps
        ttplot[:,iplot] = np.copy(tt)      # record tt(i) for plotting
        tplot[iplot] = (istep+1)*tau       # Record time for plots
        iplot += 1

methodimage = TG(x, -1, t, kappa, 1) + TG(x, 0, t, kappa, 1) + TG(x, 1, t, kappa, 1) 

plt.figure(1)
plt.plot(xplot, methodimage, label='Method of Images')
plt.title('Method of Images and FTCS @ t = 0.03')
plt.xlabel('x')
plt.ylabel('T(x,t)')
plt.xlim([-0.4, 0.4])
plt.plot(xplot, ttplot[:,-1], label='FTCS')
plt.legend()


#* Plot temperature versus x and t as a wire-mesh plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot, xplot)
ax.plot_surface(Tp, Xp, ttplot, rstride=2, cstride=2, cmap=cm.gray)
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_zlabel('T(x,t)')
ax.set_title('Diffusion of a delta spike')
plt.show()

#* Plot temperature versus x and t as a contour plot

levels = np.linspace(0., 10., num=20) 
ct = plt.contour(tplot, xplot, ttplot, levels) 
plt.clabel(ct, fmt='%1.2f') 
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Temperature contour plot')
plt.show()

