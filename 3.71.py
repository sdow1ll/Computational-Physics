import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

E = np.array([0,1,0])
B = np.array([0,0,1])

#Let q = charge of an electron
q = 1.6e-19 #C
mass = 9.1e-31 #mass of electron

time = 0
steps = 1000
tau = 10e-14

pos = np.array([0, 0, 0])
vel = np.array([0, 1, 0])

xpos = np.empty(steps)
ypos = np.empty(steps)
zpos = np.empty(steps)
timeArray = np.empty(steps)

for iters in range(steps):
    accel = q*( E + np.cross(vel, B) ) / mass #numerator term is lorenz force
    vel = vel  + accel * tau
    pos = pos + vel * tau
    xpos[iters] = pos[0] #splitting up pos array into corresponding cartesian coord.
    ypos[iters] = pos[1]
    zpos[iters] = pos[2]
    time = time + tau #kept this here to see how position changed in time
    timeArray[iters] = time

plt.title('Electron Motion in Uniform Electromagnetic Field')
plt.grid(b=None, which='major', axis='both')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xpos, ypos, 'r')
    
