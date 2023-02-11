import numpy as np
import matplotlib.pyplot as plt

#N = np.array([50, 512, 4096])
#fs = np.array([0.2, 0.2123, 0.8])
N0 = 50
fs0 = 0.2
phase = 0
tau = 1.0

t0 = np.arange(N0)*tau
y0 = np.empty(N0)

thetaj0 = np.empty(N0)

for j in range(N0):
    thetaj0[j] = 2*np.pi*fs0*j*tau % 2*np.pi 
    
for i in range(N0):
    if 0 <= thetaj0[i] < np.pi:
        y0[i] = 1
    else:
        y0[i] = -1

f0 = np.arange(N0)/(N0*tau)
ytransform0 = np.fft.fft(y0)

plt.figure(1)
plt.grid()
#plt.plot(t0,y0)
#plt.title('Original time series')
#plt.xlabel('Time')
plt.plot(f0,np.real(ytransform0),'-',f0,np.imag(ytransform0),'--')
plt.legend(['Real','Imaginary  '])
plt.title('Fourier transform N = 50 ; fs = 0.2')
plt.xlabel('Frequency')
plt.show()

plt.figure(2)
plt.grid()
powspec0 = np.empty(N0)
for i in range(N0):
    powspec0[i] = abs(ytransform0[i])**2
plt.semilogy(f0,powspec0,'-')
plt.title('Power spectrum (unnormalized) N = 50 ; fs = 0.2')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()
#-----------------------------------------------------------------------------
N1 = 512
fs1 = 0.2123

t1 = np.arange(N1)*tau
y1 = np.empty(N1)

thetaj1 = np.empty(N1)

for j in range(N1):
    thetaj1[j] = 2*np.pi*fs1*j*tau % 2*np.pi 
    
for i in range(N1):
    if 0 <= thetaj1[i] < np.pi:
        y1[i] = 1
    else:
        y1[i] = -1

f1 = np.arange(N1)/(N1*tau)
ytransform1 = np.fft.fft(y1)

plt.figure(3)
plt.grid()
#plt.plot(t1,y1)
#plt.title('Original time series')
#plt.xlabel('Time')
plt.plot(f1,np.real(ytransform1),'-',f1,np.imag(ytransform1),'--')
plt.legend(['Real','Imaginary  '])
plt.title('Fourier transform N = 512 ; fs = 0.2123')
plt.xlabel('Frequency')
plt.show()

plt.figure(4)
plt.grid()
powspec1 = np.empty(N1)
for i in range(N1):
    powspec1[i] = abs(ytransform1[i])**2
plt.semilogy(f1,powspec1,'-')
plt.title('Power spectrum (unnormalized) N = 512 ; fs =  0.2123')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()
#-----------------------------------------------------------------------------
N2 = 4096
fs2 = 0.8

t2 = np.arange(N2)*tau
y2 = np.empty(N2)

thetaj2 = np.empty(N2)

for j in range(N2):
    thetaj2[j] = 2*np.pi*fs2*j*tau % 2*np.pi 
    
for i in range(N2):
    if 0 <= thetaj2[i] < np.pi:
        y2[i] = 1
    else:
        y2[i] = -1

f2 = np.arange(N2)/(N2*tau)
ytransform2 = np.fft.fft(y2)

plt.figure(5)
plt.grid()
#plt.plot(t2,y2)
#plt.title('Original time series')
#plt.xlabel('Time')
plt.plot(f2,np.real(ytransform2),'-',f2,np.imag(ytransform2),'--')
plt.legend(['Real','Imaginary  '])
plt.title('Fourier transform N = 4096 ; fs = 0.8')
plt.xlabel('Frequency')
plt.show()

plt.figure(6)
plt.grid()
powspec2 = np.empty(N2)
for i in range(N2):
    powspec2[i] = abs(ytransform2[i])**2
plt.semilogy(f2,powspec2,'-')
plt.title('Power spectrum (unnormalized) N = 4096 ; fs = 0.8')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()