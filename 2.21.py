import numpy as np
import matplotlib.pyplot as plt

#* Set initial position and velocity of pendulum
theta0 = 170
theta = theta0 * np.pi /180     # Convert angle to radians
omega = 0.0                     # Set the initial velocity

#* Set the physical constants and other variables
g_over_L = 1.0            # The constant g/L
time = 0.0                # Initial time
irev = 0                  # Used to count number of reversals
tau = 0.02
Td = 0.2
A0 = 100

#* Take one backward step to start Verlet
adt = A0 * np.sin(2 * np.pi * (time/Td))
accel = -(g_over_L + adt) * np.sin(theta)    
theta_old = theta - omega*tau + 0.5*accel*tau**2    

#* Loop over desired number of steps with given time step
#    and numerical method
nstep = 1000
t_plot = np.empty(nstep)
th_plot = np.empty(nstep)
period = np.empty(nstep)   
for istep in range(nstep):  

    #* Record angle and time for plotting
    t_plot[istep] = time            
    th_plot[istep] = theta * 180 / np.pi  # Convert angle to degrees
    time = time + tau
  
    #* Compute new position and velocity using 
    #    Euler or Verlet method
    adt = A0 * np.sin(2 * np.pi * (time/Td))
    accel = (-g_over_L + adt) * np.sin(theta)   # Gravitational acceleration
    theta_new = 2*theta - theta_old + tau**2 * accel
    theta_old = theta               # Verlet method
    theta = theta_new  
  
    #* Test if the pendulum has passed through theta = 0;
    #    if yes, use time to estimate period
    if theta*theta_old < 0 :  # Test position for sign change
        print('Turning point at time t = ',time)
        if irev == 0 :          # If this is the first change,
            time_old = time     # just record the time
        else:
            period[irev-1] = 2*(time - time_old)
            time_old = time
        irev = irev + 1     # Increment the number of reversals

# Estimate period of oscillation, including error bar
nPeriod = irev-1    # Number of times the period was measured
AvePeriod = np.mean( period[0:nPeriod] )
ErrorBar = np.std(period[0:nPeriod]) / np.sqrt(nPeriod)
print('Average period = ', AvePeriod, ' +/- ', ErrorBar)

# Graph the oscillations as theta versus time
plt.plot(t_plot, th_plot, 'r-')
plt.title('Driven Pendulum Motion')
plt.xlabel('Time')
plt.ylabel(r'$\theta$ (degrees)')
plt.show()

#When we look at the figure, it shows that at 180 degrees the oscillator is most stable. There are
#no extra "wiggles" like you see when the oscillator is at 200 degrees or 160 degrees.

