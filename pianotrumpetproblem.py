import numpy as np
import matplotlib.pyplot as plt

piano = open('piano.txt')
trumpet = open('trumpet.txt')
piano.close()
trumpet.close()

pianodata = np.loadtxt('piano.txt')
trumpetdata = np.loadtxt('trumpet.txt')


N = len(pianodata)
tau = 1/44100
t = np.arange(N)*tau  
f = np.arange(N)/(N*tau)

pianotransform = np.fft.fft(pianodata)
trumpettransform = np.fft.fft(trumpetdata)

plt.figure(1)
plt.grid()
plt.plot(t, pianodata)
plt.title('Waveform of piano')
plt.xlabel('Time')
plt.ylabel('Sound')

plt.figure(2)
plt.grid()
plt.plot(f[0:5000],np.real(pianotransform[0:5000]),'-')
plt.title('FFT for piano')
plt.xlabel('Freq. (Hz)')
plt.ylabel('Amplitude')

plt.figure(3)
plt.grid()
plt.plot(t, trumpetdata)
plt.title('Waveform of trumpet')
plt.xlabel('Time')
plt.ylabel('Sound')

plt.figure(4)
plt.grid()
plt.plot(f[0:5000],np.real(trumpettransform[0:5000]),'-')
plt.title('FFT for trumpet')
plt.xlabel('Freq. (Hz)')
plt.ylabel('Amplitude')

pianopeak = np.argmax(np.real(pianotransform[0:5000])) #i know this is where my fourier transform spikes.
trumpetpeak = np.argmax(np.real(trumpettransform[0:5000]))

#print(np.indexof(np.amax(np.real(pianotransform[0:5000]))))

print('Piano frequency:', f[1191], 'Hz')
print('Trumpet frequency:', f[2367], 'Hz')
print('Piano note occurs at C5 and trumpet note occurs at C6. These are the same note, but the trumpet is at a higher octave than the piano.')
print('')




