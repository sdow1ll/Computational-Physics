import numpy as np
import matplotlib.pyplot as plt

#* Set initial position and velocity of pendulum
omega = 0.0                     # Set the initial velocity

#* Set the physical constants and other variables
g_over_L = 1.0            # The constant g/L
time = 0.0                # Initial time
irev = 0                  # Used to count number of reversals
tau = 0.01
irev = 0

theta0 = np.linspace(0,180,181)
accel = np.zeros((1,181))
theta = np.array([])

for i in range(len(theta0)):  
    theta = np.insert(theta, 0, [theta0[i] * np.pi /180])
theta = theta[::-1] 

#COMPUTED PERIOD VS. THETA:
accel = - g_over_L * np.sin(theta)
theta_old = theta - omega*tau + 0.5*accel*tau**2
th_plot = np.empty(181)
nstep = 180
period = np.empty(nstep)   # Used to record period estimates

for istep in range(nstep):
    th_plot = theta[istep] * 180 / np.pi
    time = time + tau
    accel = - g_over_L * np.sin(theta)
    theta_new = 2*theta - theta_old + tau**2 * accel
    theta_old = theta
    theta = theta_new
    
    if theta[i]*theta_old[i] < 0:
        if irev == 0:
            time_old = time
        else:
            period[irev-1] = 2*(time -time_old)
            time_old = time
        irev = irev + 1

periodArray = np.empty(181)
for i in range(len(theta0)):
    nPeriod = irev-1
    AvePeriod = np.mean( period[0:nPeriod] )
    periodArray[i] = AvePeriod
    
#FIRST PERIOD APPROX:
T1array = np.empty(181)
for i in range(181):
    T1array[i] = 2 * np.pi * np.sqrt(1/g_over_L)

#SECOND PERIOD APPROX:
T2array = np.empty(180)
def T2(thetam):
    approximation = 2*np.pi * (1 + (1/16 * thetam))
    return approximation

# Graph the oscillations as theta versus time
plt.plot(theta0, periodArray, '+')
plt.plot(theta0, T1array, 'r')
plt.plot(theta0, T2(theta0), 'b')
plt.xlabel('Theta')
plt.ylabel('Period')
plt.title('Pendulum Oscillations')
plt.legend(['Computed', '1st Approx', '2nd Approx'])
plt.show()

