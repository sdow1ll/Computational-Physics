#import matplotlib.pyplot as plt
import numpy as np

def interd(xi, x, y):
    
    c1 = y[0] / ( (x[0] - x[1]) * (x[0] - x[2]) )
    c2 = y[1] / ( (x[1] - x[0]) * (x[1] - x[2]) )
    c3 = y[2] / ( (x[2] - x[0]) * (x[2] - x[1]) )
    
     #deriviative of lagrange polynomial:
    dyi = (( 2*xi - x[2] - x[1] ) * c1 ) + ( (2*xi - x[2] - x[0]) * c2 ) + ((2*xi - x[1] - x[0]) * c3)
    return dyi

#using 1.12 as a test for my interpolating derivative 
x = np.array([0, 0.5, 1])
J0 = np.array([1, 0.9385, 0.7652])
xi = np.array([0.3, 0.9, 1.1, 1.5, 2.0])

for i in xi:
    print(interd(i, x, J0))