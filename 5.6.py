import numpy as np
import matplotlib.pyplot as plt

def linreg(x,y,sigma):
    """Function to persform linear regression (fit a line)
       Inputs
        x       Independent variable
        y       Dependent variable
        sigma   Estimated error in y
       Outputs
        a_fit   Fit parameters; a(1) is intercept, a(2) is slope
        sig_a   Estimated error in the parameters a()
        yy      Curve fit to the data
        chisqr  Chi squared statistic
    """
    
    #* Evaluate various sigma sums
    s = 0.; sx = 0.; sy = 0.; sxy = 0.; sxx = 0.
    for i in range(len(x)):
        sigmaTerm = sigma[i]**(-2)
        s += sigmaTerm              
        sx += x[i] * sigmaTerm
        sy += y[i] * sigmaTerm
        sxy += x[i] * y[i] * sigmaTerm
        sxx += x[i]**2 * sigmaTerm
    denom = s*sxx - sx**2

    #* Compute intercept a_fit(1) and slope a_fit(2)
    a_fit = np.empty(2)
    a_fit[0] = (sxx*sy - sx*sxy)/denom
    a_fit[1] = (s*sxy - sx*sy)/denom

    #* Compute error bars for intercept and slope
    sig_a = np.empty(2)
    sig_a[0] = np.sqrt(sxx/denom)
    sig_a[1] = np.sqrt(s/denom)

    #* Evaluate curve fit at each data point and compute Chi^2
    yy = np.empty(len(x))
    chisqr = 0.
    for i in range(len(x)):
        yy[i] = a_fit[0] + a_fit[1]*x[i]          # Curve fit to the data
        chisqr += ( (y[i]-yy[i])/sigma[i] )**2    # Chi square
    return yy
def pollsf(x, y, sigma, M):
    """Function to fit a polynomial to data
       Inputs 
        x       Independent variable
        y       Dependent variable
        sigma   Estimate error in y
        M       Number of parameters used to fit data
       Outputs
        a_fit   Fit parameters; a(1) is intercept, a(2) is slope
        sig_a   Estimated error in the parameters a()
        yy      Curve fit to the data
        chisqr  Chi squared statistic
    """
    
    #* Form the vector b and design matrix A   
    N = len(x)
    b = np.empty(N)
    A = np.empty((N,M))
    for i in range(N):
        b[i] = y[i]/sigma[i]
        for j in range(M):
            A[i,j] = x[i]**j / sigma[i] 

    #* Compute the correlation matrix C 
    C = np.linalg.inv( np.dot( np.transpose(A), A) )

    #* Compute the least squares polynomial coefficients a_fit
    a_fit = np.dot(C, np.dot( np.transpose(A), np.transpose(b)) )

    #* Compute the estimated error bars for the coefficients
    sig_a = np.empty(M)
    for j in range(M):
        sig_a[j] = np.sqrt(C[j,j])

    #* Evaluate curve fit at each data point and compute Chi^2
    yy = np.zeros(N)
    chisqr = 0.
    for i in range(N):
        for j in range(M):
            yy[i] += a_fit[j]*x[i]**j   # yy is the curve fit
        chisqr += ((y[i]-yy[i]) / sigma[i])**2
        
    return a_fit

day = np.array([1, 2, 3, 4, 5])
DJA = np.array([2470, 2510, 2410, 2350, 2240])

sigma = np.empty(len(day)) #constant err bars
sigma.fill(1)

poly2coeff = np.transpose(pollsf(day,DJA,sigma,3))
poly3coeff = np.transpose(pollsf(day, DJA, sigma, 4))
poly4coeff = np.transpose(pollsf(day, DJA, sigma, 5))


def poly2(x, a0, a1, a2):
    return a2*x**2 + a1*x + a0

def poly3(x, a0, a1, a2, a3):
    return a3*x**3 + a2*x**2 + a1*x + a0

def poly4(x, a0, a1, a2, a3, a4):
    return a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0

xmesh = np.linspace(0,5,100)
plt.plot(day, DJA, 'r*', label='data pts')
plt.plot(day, linreg(day,DJA,sigma),label='1st Deg')
plt.plot(xmesh, poly2(xmesh, poly2coeff[0], poly2coeff[1], poly2coeff[2]), label='2nd Deg')
plt.plot(xmesh, poly3(xmesh, poly3coeff[0], poly3coeff[1], poly3coeff[2], poly3coeff[3]),label='3rd Deg')
plt.plot(xmesh, poly4(xmesh, poly4coeff[0], poly4coeff[1], poly4coeff[2], poly4coeff[3], poly4coeff[4]),label='4th Deg')
plt.title('Dow Jones Averages')
plt.xlabel('Days')
plt.ylabel('DJA')
plt.legend()
print('Expected value for Day 6:', poly4(6, poly4coeff[0], poly4coeff[1], poly4coeff[2], poly4coeff[3], poly4coeff[4]))


