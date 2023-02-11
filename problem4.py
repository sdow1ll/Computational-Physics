import numpy as np
import matplotlib.pyplot as plt

r = 1

randangle = np.zeros([1000,2])

def theta(rand):
    return np.arccos(1 - 2*rand)

def phi(rand):
    return 2*np.pi*rand

for i in range(len(randangle)):
    randangle[i,0] = theta(np.random.random())

    randangle[i,1] = phi(np.random.random())
    
    
plt.figure(1)
plt.hist(randangle[:,0])
plt.title('theta distribution: 0.5sin(theta)')
plt.figure(2)
plt.title('phi distribution: 1/2pi')
plt.hist(randangle[:,1])

theta = randangle[:,0]
phi = randangle[:,1]

x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(phi)

cartesian = np.transpose(np.array([x, y, z]))

#converting angles into latitude and longitude:
latitude = (theta[0]*180/np.pi) 
longitude = (phi[0]*180/np.pi) 

print(latitude, longitude)

#in my first trial i got kazakhstan. the latitude and longitude 
#will change whenever you run the program though.
#these were my coordinates: 52.750248785083556, 73.17241519634857

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Random points around the world')
ax.scatter(x, y, z, c='green', marker='^', s=2)


