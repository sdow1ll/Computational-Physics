import numpy as np
import matplotlib.pyplot as plt 

#a
fig1 = plt.figure(figsize=(5, 5), dpi=300)
x = np.arange(0,20, 0.05)
ya = np.exp(-x/4)*np.sin(x)
plt.plot(x, ya)
plt.xticks([0, 5, 10, 15, 20])
plt.title('f(x) = exp(-x/4)*sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(0,20)
plt.ylim(-0.4, 1)
plt.grid(1, which='major', axis='both')
plt.show()
#b
fig2 = plt.figure(figsize=(5, 5), dpi=300)
xa = np.arange(0,20, 0.05)
ya = np.exp(-xa/4)*np.sin(xa)
xb = np.array([0, 1.5, 3, 4.8, 6.6, 7.7, 8.7, 10.8, 13.2, 14.3, 16, 17, 18])
yb = np.exp(-xb/4)*np.sin(xb)
plt.plot(xa,ya)
plt.plot(xb, yb, 'o')
yc = np.exp(-x/4)
plt.plot(xa, yc, ls='dashed')
plt.annotate('exp(-x/4)', xy=[5,0.35])
plt.xticks([0, 5, 10, 15, 20])
plt.title('f(x) = exp(-x/4)*sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(0,20)
plt.ylim(-0.4, 1)
plt.show()

#d (I didn't realize we didn't need to do this until later, 
#  but I wanted to keep it because I was proud of myself.)
rad = np.arange(0,  2*(np.pi), 0.01)  

ax = plt.axes(projection="polar")
ax.plot(rad, np.cos(6 * rad + (3*np.pi)))
ax.set_yticks([0.5])
ax.set_rlabel_position(80)
plt.thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], labels=['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330'])

