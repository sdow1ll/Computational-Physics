import numpy as np
import matplotlib.pyplot as plt

Mauna = open('Mauna.txt')
Barrow = open('Barrow.txt')
Mauna.close()
Barrow.close()

M = np.loadtxt('Mauna.txt')
B = np.loadtxt('Barrow.txt')

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
    return [yy, a_fit]

Msigma = np.empty(len(M)) #constant err bars
Msigma.fill(0.16)
Bsigma = np.empty(len(B))
Bsigma.fill(0.27)

yearM = np.linspace(1980, 1990, len(M))
yearB = np.linspace(1980, 1990, len(B))

CO2M = linreg(yearM, M, Msigma)[0]
CO2B = linreg(yearB, B, Bsigma)[0]

Maunaparam = linreg(yearM, M, Msigma)[1]
Barrowparam = linreg(yearB, B, Bsigma)[1]

print('Rate of increase of CO2 for Mauna:', Maunaparam[1])
print('Rate of increase of CO2 for Barrow:', Barrowparam[1])

plt.figure(1)
plt.title('Mauna CO2 Emissions')
plt.plot(yearM, M, 'rx')
plt.ylabel('CO2 ppm')
plt.xlabel('Year')
plt.plot(yearM, CO2M)



plt.figure(2)
plt.title('Barrow CO2 Emissions')
plt.ylabel('CO2 ppm')
plt.xlabel('Year')
plt.plot(yearB, B, 'gx')
plt.plot(yearB, CO2B)



year = np.arange(2021, 2100, 1)

def linfunc(year, param):
    return param[0] + param[1]*year

param = linreg(yearM, M, Msigma)[1]
print('CO2 emissions for 2021:', linfunc(2021, param))

yrArray = np.empty(len(year))

for i in range(len(year)):
    yrArray[i] = linfunc(year[i], param)        
print('It takes', int(np.where((434 < yrArray) & (yrArray < 435))[0]), 'yrs for CO2 emissions to reach 10% above 2021 emissions') 
print('Therefore, by 2050 the CO2 emissions will increase by 10%')
print('CO2 emissions for 2050:', yrArray[29])











