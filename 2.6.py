import numpy as np
import matplotlib.pyplot as plt

#* Set initial position and velocity of the baseball

r01 = np.array([0,50])
r0100 = np.array([3, 50]) # Initial vector position (in meters)
speed = 0 # Initial speed (in m/s)
theta = 0 # Initial angle (in degrees)

v0 = np.array([speed * np.cos(theta*np.pi/180), 
      speed * np.sin(theta*np.pi/180)])      # Initial vector velocity (m/s)

r100 = np.copy(r0100)   # Set initial position 
v100 = np.copy(v0)   # Set initial velocity
r1 = np.copy(r01)
v1 = np.copy(v0)

#* Set physical parameters (mass, Cd, etc.)
Cd = 0.5      
area = 4.3e-3  
grav = 9.81   

mass100 = 45.3592
mass1 = 0.4535 

rho = 1.2  
    
air_const100 = -0.5*Cd*rho*area/mass100   
air_const1 = -0.5*Cd*rho*area/mass1

#* Loop until ball hits ground or max steps completed
tau = 0.05   # timestep (sec)
maxstep = 1000 #Maximum number of steps

# Initialize empty arrays to store the trajectory

xplot100 = np.empty(maxstep)
yplot100 = np.empty(maxstep)
xplot1 = np.empty(maxstep)
yplot1 = np.empty(maxstep)

#100 lb ball:
for istep in range(maxstep):

    #* Record position (computed and theoretical) for plotting
    xplot100[istep] = r100[0]   # Record trajectory for plot for 100lb
    yplot100[istep] = r100[1]
  
    accel100 = air_const100 * np.linalg.norm(v100) * v100
    accel100[1] = accel100[1] - grav                    
  
    #* Calculate the new position and velocity using Euler method
    r100 = r100 + tau*v100                    # Euler step
    v100 = v100 + tau*accel100
  
    if r100[1] < 0 : 
        laststep100 = istep + 1
        
        xplot100[laststep100] = r100[0]  
        yplot100[laststep100] = r100[1]
        break          

for istep in range(maxstep):
    xplot1[istep] = r1[0]   # Record trajectory for plot for 100lb
    yplot1[istep] = r1[1]
    t = istep*tau         # Current time
  
    accel1= air_const1 * np.linalg.norm(v1) * v1   
    accel1[1] = accel1[1] - grav               
  
    r1 = r1 + tau*v1                
    v1 = v1 + tau*accel1
  
    #* If ball reaches ground (y<0), break out of the loop
    if r1[1] < 0 : 
        laststep1 = istep + 1

        xplot1[laststep1] = r1[0]
        yplot1[laststep1] = r1[1]
        break                   # Break out of the for loop
#* Print maximum range and time of flight
print('Max height for 100lb ball:', r100[1], 'meters') 
print('Max height for 1lb ball:', r1[1], 'meters')
print('Flight time for 100lb ball:', laststep100*tau , ' seconds')
print('Flight time for 1lb ball:', laststep1*tau, 'seconds')
print('Difference of ball heights:', np.abs(np.subtract(r100[1], r1[1])), 'm')


plt.style.use('classic')
plt.xlim([-1,5])
plt.ylim([-5, 60])
plt.grid(b=None, which='major', axis='both')
plt.plot(xplot100[:], yplot100[:], 'rx')
plt.plot(xplot1[:], yplot1[:], 'b+', label='1lb ball')
plt.show()


plt.legend(['100 Ball', '1lb Ball']);
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.title('Projectile motion')
plt.show()


#from the script file, it shows that the 100lb ball will hit the ground first. 
#the balls are 0.7485122977801524 m apart which is not equilvalent to 
#2 inches at all. Thus, Galileo was wrong. 