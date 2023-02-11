#Exam 1 Problem 3:

import numpy as np
import matplotlib.pyplot as plt

def wienfunc(x):
    f = 5 * np.exp(- x) + x - 5
    return f

def wienderiv(x):
    fprime = -5 * np.exp(- x)
    return fprime

x0 = 5
tol = 1e-10


def bisection(a,b,tol):
    xleft = a
    xright = b
    counter = 0
    while np.abs(xleft - xright) >= tol:
            c = (xleft + xright)/2
            result = wienfunc(xleft) * wienfunc(c)
            if result > tol:
                xleft = c
            else:
                if result < tol:
                    xright = c
            counter+=1
            print('Iterations:', counter)
            print(c)
    return c

#print(bisection(-1, 1, tol)) #zero root
print(x)
                
#wien's displacement  constant is b = hc/kx and lambda = b/T

#I got all these values from wikipedia
h = 6.62607015e-34 #joules * seconds
c = 3e10 # m/s
k = 1.380649e-23 #joules * kelvin^-1
x = bisection(3, 6, tol)
wienb = h*c / k*x

print('')
print("Wien's b constant is,", wienb)

wavelength = 502e-9 #m
T = wienb/wavelength

print('Temperature of the Sun in K:', T, 'K')