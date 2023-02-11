import numpy as np
import scipy.special


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

x = np.array([0, 0.5, 1])
J0 = np.array([1, 0.9385, 0.7652])
xi = np.array([0.3, 0.9, 1.1, 1.5, 2.0])




for i in xi:    
    print(intrpf(i, x, J0))
print('')
for i in xi:
    print(scipy.special.jn(0, i))

# My intrpf() function is close to the scipy.special.jn() method, but slightly off
# in the 10^-3 digit and further also for the other values of x and J0.