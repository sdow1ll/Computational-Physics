import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt

#* Set initial position and velocity of the

r0 = np.array([0,0]) # Initial vector position (in meters)
speed = 50 # Initial speed (in m/s)
theta = 45 # Initial angle (in degrees)
    
v0 = np.array([speed * np.cos(theta*np.pi/180), 
      speed * np.sin(theta*np.pi/180)])      # Initial vector velocity (m/s)

r = np.copy(r0)   # Set initial position 
v = np.copy(v0)   # Set initial velocity

#* Set physical parameters (mass, Cd, etc.)
Cd = 0.35      # Drag coefficient (dimensionless)
area = 4.3e-3  # Cross-sectional area of projectile (m^2)
grav = 9.81   # Gravitational acceleration (m/s^2)
mass = 0.145   # Mass of projectile (kg)

airFlag = 0 # Set whether to incorporate air resistance (1) or not (0)
#problem says to NOT incorporate air res in computation

if airFlag == 0:
    rho = 0.      # No air resistance
else:
    rho = 1.2  # Density of air (kg/m^3)
    
air_const = -0.5*Cd*rho*area/mass   # Air resistance constant (0 if airFlag = 0)

#* Loop until ball hits ground or max steps completed
tau = 0.01   # timestep (sec)
maxstep = 723 #Maximum number of steps

# Initialize empty arrays to store the trajectory

xplot = np.empty(maxstep)
yplot = np.empty(maxstep)
xNoAir = np.empty(maxstep)
yNoAir = np.empty(maxstep)

for istep in range(maxstep):

    #* Record position (computed and theoretical) for plotting
    xplot[istep] = r[0]   # Record trajectory for plot
    yplot[istep] = r[1]
    t = istep*tau         # Current time
    xNoAir[istep] = r0[0] + v0[0]*t
    yNoAir[istep] = r0[1] + v0[1]*t - 0.5*grav*t**2
  
    #* Calculate the acceleration vector of the ball 
    accel = air_const * np.linalg.norm(v) * v   # Air resistance
    accel[1] = accel[1] - grav                  # Gravity
  
    #* Calculate the new position and velocity using Euler method
    r = r + tau*v                    # Euler step
    v = v + tau*accel     
  
    #* If ball reaches ground (y<0), break out of the loop
    if r[1] < 0 : 
        laststep = istep+1
        xplot[laststep] = r[0]  # Record last values computed
        yplot[laststep] = r[1]
        break                   # Break out of the for loop

#* Graph the trajectory of the baseball
# Mark the location of the ground by a straight line
xground = np.array([0., xNoAir[laststep-1]])
yground = np.array([0., 0.])

#INTERPOLATION FUNCTION:
def intrpf(xi, x, y):
    """ Inputs
        x    Vector of x coordinates of data points (3 values)
        y    Vector of y coordinates of data points (3 values)
        xi   The x value where interpolation is computed
      Output
        yi   The interpolation polynomial evaluated at xi
    """
    # lagrange polynomial:
    yi = ( (xi-x[1])*(xi-x[2])/((x[0]-x[1])*(x[0]-x[2])) * y[0]
    + (xi-x[0])*(xi-x[2])/((x[1]-x[0])*(x[1]-x[2])) * y[1]
    + (xi-x[0])*(xi-x[1])/((x[2]-x[0])*(x[2]-x[1])) * y[2] )
    return yi

nplot = laststep #(576 STEPS)
xpoints = np.empty(nplot)
ypoints = np.empty(nplot)
x = np.array([xplot[laststep - 3], xplot[laststep - 2], xplot[laststep - 1]])
y = np.array([yplot[laststep - 3], yplot[laststep - 2], yplot[laststep - 1]])
xr = np.array([0, 255]) #255 is last Euler data point which goes over the ground

for i in range(nplot) :
    xpoints[i] = xr[0] + (xr[1]-xr[0])* i/float(nplot)
    ypoints[i] = intrpf(xpoints[i], x, y)

plt.xlim(0,260)
plt.plot(xpoints[-3:], ypoints[-3:], 'o')
plt.plot(xplot[0:laststep+1], yplot[0:laststep+1], xground, yground)
plt.ylabel('Height (m)')
plt.xlabel('Range (m)')
plt.legend(['Interpolation', 'Euler Method', 'Ground'])
plt.title('Euler Method & Interpolation w/o Air Res (tau = 0.01)')

#* Print maximum range and time of flight

print('For tau = 0.01,')
print('Max range for Euler Method:', r[0], 'meters') 
print('Flight time for Euler Method:', laststep*tau , ' seconds')
print('Max range of interpolated point:', xpoints[-1], 'm')
print('Flight time for Interpolation:', 721*tau, 'seconds' )
print('Corrected range value of interpolation and Euler Method:', r[0] - xpoints[-1],  'm')
print('Corrected time value of interpolation and Euler Method:',  laststep*tau 
      - 721*tau,  'seconds')

print('')
print('')

print('For tau = 0.1,')
print('Max range for Euler Method: 261.62950903902265 meters') 
print('Flight time for Euler Method: 7.4 seconds')
print('Max range of interpolated point: 257.47297297297297 m')
print('Flight time for Interpolation: ￼￼￼7.30000000001  seconds' )
print('Corrected range value of interpolation and Euler Method: 4.156536066049682 m')
print('Corrected time value of interpolation and Euler Method: 0.09999999999999964 seconds')
