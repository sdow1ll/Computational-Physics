#6.4
import numpy as np
import matplotlib.pyplot as plt

t = 0.03
kappa = 1
L = 1
x = np.linspace(-1.5, 1.5, 100)

def TG(x, n, t, kappa):
    x0 = n*L
    output = np.empty(len(x))
    for i in range(len(x)):    
        output[i] = exp(-((x[i]-x0)**2) / (4*kappa*t)) / (((2*kappa*t)**(1/2)) * (2*np.pi)**(1/2))
    return output


negone = -TG(x, -1, t, kappa)
zero = TG(x, 0, t, kappa)
one = -TG(x, 1, t, kappa)
yzero = np.zeros(100)

plt.figure(1)
plt.plot(x, yzero, ls='--')
plt.plot(x, zero)
plt.ylim([-2, 2])
plt.plot(x, negone+one, ls='--')
plt.xlabel('x/L')
plt.ylabel('T(x,t)')

plt.figure(2)
plt.vlines(-0.5, -3, 3, ls='--')
plt.vlines(0.5, -3, 3, ls='--')
plt.ylim([-2, 2])
plt.plot(x, zero+negone+one)
