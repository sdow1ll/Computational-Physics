import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#* Set initial position and velocity of the comet.
r0 = 1.0
v0 = 3*np.pi / 2

r = np.array([r0, 0])
v = np.array([0, v0])


#* Set physical parameters (mass, G*M)
GM = 4 * np.pi**2      # Grav. const. * Mass of Sun (au^3/yr^2)
mass = 1.0             # Mass of comet 
adaptErr = 1.0e-3      # Error parameter used by adaptive Runge-Kutta
time = 0.0

nStep = 200
tau = 0.005

rplot = np.empty(nStep)           
thplot = np.empty(nStep)
tplot = np.empty(nStep)
kinetic = np.empty(nStep)
potential = np.empty(nStep)
rArrayx = np.empty(nStep) #I made this array to track the value of r in for loop
rArrayy = np.empty(nStep)

for iStep in range(nStep):  
    #* Record position and energy for plotting.
    tplot[iStep] = time 
    time += tau
    rplot[iStep] = np.linalg.norm(r)  # Record position for polar plot
    thplot[iStep] = np.arctan2(r[1],r[0]) 
    kinetic[iStep] = .5*mass*np.linalg.norm(v)**2   # Record energies
    potential[iStep] = - GM*mass/np.linalg.norm(r)
    accel = -GM*r/np.linalg.norm(r)**3   
    v = v + tau * accel
    r = r + tau*v              # Euler-Cromer step  
    rArrayx[iStep] = r[0]
    rArrayy[iStep] = r[1]
    if rplot[iStep] == rplot[115]: #at iteration 115 the comet completes its cycle.
        break
    
#* Graph the trajectory of the comet.
trajectory = plt.figure(1)
plt.subplot(111, projection='polar')  # Use polar plot for graphing orbit
plt.plot(thplot[:116],rplot[:116],'+')  
plt.title('Distance (AU)') 
trajectory.show()

#* Graph the energy of the comet versus time.
totalE = kinetic + potential   # Total energy
energy = plt.figure(2)
plt.plot(tplot[:116], kinetic[:116], tplot[:116], potential[:116], tplot[:116], totalE[:116])
plt.legend(['Kinetic','Potential','Total']);
plt.xlabel('Time (yr)')
plt.ylabel(r'Energy ($M AU^2/yr^2$)')
energy.show()


#a)
print('a)')
a = max(rArrayx[0:116]) - min(rArrayx[0:116]) # take x data points in rArray and find max/min length
#of plot. I had to do this because the orbit is not centered on the polar plot grid.
#found on wikipedia page:
#https://en.wikipedia.org/wiki/Semi-major_and_semi-minor_axes
print('Semimajor axis distance:', a, 'AU')

b = max(rArrayy[0:116]) - min(rArrayy[0:116])
print('Semiminor axis distance:', b, 'AU')

T = (( 4 * (np.pi ** 2) ) / GM ) * a**3
print('Period of orbit:', T, 'yrs')

eccentricity = sqrt(1 - (b**2 / a**2))
print('Eccentricity of orbit:', eccentricity)

L = np.cross(r, mass*v)

otherEccentricity = sqrt((1 + (2*totalE[0]*(L**2) / (GM ** 2 * mass**3))))
print("Other Eccentricity:", otherEccentricity)
#eccentricity I have to compare.

perihelion = a*(1 - eccentricity)
print('Perihelion distance:', perihelion, 'AU')

#b)
#we can show Kepler's 3rd law works by looking at the energy plot.
#when kinetic energy is highest, we know that velocity will increase. the rate of change
#for kinetic energy takes a short amount of time from 0.2 to 0.3 yrs. Therefore, Kepler's 3rd law
#is valid.

#c)
print('c)')
kineticAvg = np.mean(kinetic[:116])
potentialAvg = np.mean(potential[:116])

Krelationship = - 0.5 * potentialAvg
print('Kinetic E eqn:', Krelationship)
print('Kinetic Avg:', kineticAvg)

#While kinetic E eqn and kinetic avg are not exactly the same, they are substantially close.
#computational error comes into play here, but when we see the plot for totalE we observe that
#the energies are conserved.
