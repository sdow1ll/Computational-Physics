import numpy as np
import matplotlib.pyplot as plt

sunspot = open('sunspots.txt')
sunspot.close()

sun = np.loadtxt('sunspots.txt', delimiter=',')
plt.figure(1)
#plt.plot(sun[612:875,0], sun[612:875,2]) #shows one period of the data from 1800 - 1821

plt.figure(2)
plt.plot(sun[:,0], sun[:,2], linewidth=0.5)
plt.title('Sunspot data')
plt.xlabel('year')
plt.ylabel('no. of sunspots')

datapts = sun[:,2]

#the plot does in fact look periodic
#period of this data set seems to be approximately 11 yrs

N = len(datapts)
n = np.arange(N)
k = n.reshape(N,1)
Yk = np.dot(np.exp(-2j * np.pi * k * n / N), datapts)
tau = 12 #1 yr is the time increment b/w data points
freq = k/tau*N
plt.figure(3)
plt.plot(freq, np.abs(Yk)**2)
plt.title('Magnitude DFT Squared (original plot)')
plt.xlabel('freq. (1/months)')
plt.ylabel('|Yk|^2')

#period from ft shows that the period of the sunspot data is 11 yrs. 
plt.figure(4)
Yk1 = np.dot(np.exp(-2j * np.pi * k * n / N), datapts)
Yk1[100:] = 0
invft= np.fft.irfft(Yk, n=N)
invft1 = np.fft.irfft(Yk1, n=N)
monthsinv = np.linspace(1, 12, N)
plt.plot(sun[:,0], sun[:,2], linewidth=0.5, label='Original Data')
plt.plot(sun[:,0], invft1, label='Inverse Transform')
plt.title('Comparison of inverse w/ original')
plt.legend()

#setting the rest of the coefficients to zero smoothens the inverse fourier transform. So that
#the original time series is more readable. Also by setting the rest of the coefficients to zero,
#we don't need to focus on the other frequency signals we got from our fourier transform.

